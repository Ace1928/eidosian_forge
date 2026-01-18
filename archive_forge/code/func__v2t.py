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
def _v2t(self, vec):
    """
        Convert token indices to string of tokens.
        """
    new_vec = []
    if hasattr(vec, 'cpu'):
        vec = vec.cpu()
    for i in vec:
        if i == self.END_IDX:
            break
        elif i != self.START_IDX:
            new_vec.append(i)
    return self.dict.vec2txt(new_vec)