from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import round_sigfigs
from .modules import TransresnetModel
from parlai.tasks.personality_captions.build import build
import os
import random
import json
import numpy as np
import torch
import tqdm
@staticmethod
def add_cmdline_args(argparser):
    """
        Add command line args.
        """
    arg_group = argparser.add_argument_group('Transresnet Arguments')
    TransresnetModel.add_cmdline_args(argparser)
    argparser.add_argument('--freeze-patience', type=int, default=-1, help='How long to freeze text encoders')
    argparser.add_argument('--one-cand-set', type='bool', default=False, help='True if each example has one set of shared label candidates')
    argparser.add_argument('--fixed-cands-path', type=str, default=None, help='path to text file with candidates')
    argparser.add_argument('--pretrained', type='bool', default=False, help='True if pretrained model')
    DictionaryAgent.add_cmdline_args(argparser)
    return arg_group