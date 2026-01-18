from parlai.core.torch_agent import TorchAgent, Output, Batch
from parlai.utils.misc import round_sigfigs
from parlai.utils.torch import padded_tensor, argsort, neginf
from .modules import Seq2seq, opt_to_kwargs
from .util import ConvAI2History, show_beam_cands, reorder_extrep2gram_qn
from .controls import (
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, Counter
from operator import attrgetter
import os
import math
import json
import tempfile
import copy
def _init_controls(self):
    """
        Initialize controls.

        Sets the following:

          self.control_vars: list of strings. The CT controls sorted alphabetically.

          self.control_settings: a dictionary containing info about the CT controls.
            Each control name maps to a dictionary that contains:
              'embsize': embedding size for this control
              'num_buckets': num buckets for this control
              'set_value': a set value for this control, or None
              'idx': the index of this control in this list self.control_vars

          self.wd_features: list of strings, the WD features to use.

          self.wd_wts: list of floats, the WD weights to use.
        """
    self.control_vars = sorted(self.opt['control_vars'].split(',')) if self.opt['control_vars'] != '' else []
    ctrl_numbucket_override = {}
    if self.opt['control_num_buckets'] != '':
        ctrl_numbucket_override = {s.split(':')[0]: int(s.split(':')[1]) for s in self.opt['control_num_buckets'].split(',')}
    ctrl_esz_override = {}
    if self.opt['control_embeddingsize'] != '':
        ctrl_esz_override = {s.split(':')[0]: int(s.split(':')[1]) for s in self.opt['control_embeddingsize'].split(',')}
    set_controls = {}
    if self.opt['set_controls'] != '':
        set_controls = {}
        for s in self.opt['set_controls'].split(','):
            control, set_val = (s.split(':')[0], s.split(':')[1])
            if control not in self.control_vars:
                raise ValueError("Received --set-controls for control '%s', but that is not one of the existing CT controls for this model, which are: %s" % (control, ', '.join(self.control_vars)))
            try:
                set_val = int(set_val)
            except ValueError:
                raise ValueError("Received --set-controls '%s' for CT control '%s'. The set value must be an integer." % (set_val, control))
            set_controls[control] = int(set_val)
    self.control_settings = {}
    for idx, c in enumerate(self.control_vars):
        d = {}
        d['embsize'] = ctrl_esz_override[c] if c in ctrl_esz_override else CONTROL2DEFAULTEMBSIZE[c]
        d['num_buckets'] = ctrl_numbucket_override[c] if c in ctrl_numbucket_override else CONTROL2DEFAULTNUMBUCKETS[c]
        if c in set_controls:
            set_val = set_controls[c]
            if set_val not in range(d['num_buckets']):
                raise ValueError("Received --set-controls '%s' for CT control '%s', which has num_buckets=%i. The set value must be between 0 and %i." % (set_val, c, d['num_buckets'], d['num_buckets'] - 1))
        d['set_value'] = set_controls[c] if c in set_controls else None
        d['idx'] = idx
        self.control_settings[c] = d
    if self.opt.get('weighted_decoding', '') != '':
        if self.beam_size == 1:
            raise ValueError('WD control is not currently implemented for greedy search. Either increase --beam-size to be greater than 1, or do not enter --weighted-decoding (-wd).')
        wd_feats_wts = [(s.split(':')[0], float(s.split(':')[1])) for s in self.opt['weighted_decoding'].split(',')]
        self.wd_features = [f for f, w in wd_feats_wts]
        for wd_feat in self.wd_features:
            if wd_feat not in WDFEATURE2UPDATEFN:
                raise ValueError("'%s' is not an existing WD feature. Available WD features: %s" % (wd_feat, ', '.join(WDFEATURE2UPDATEFN.keys())))
        self.wd_wts = [w for f, w in wd_feats_wts]
    else:
        self.wd_features, self.wd_wts = ([], [])