from typing import Dict, Any, Union, List, Tuple, Optional
from abc import ABC, abstractmethod
import random
import os
import torch
import parlai.utils.logging as logging
from torch import optim
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.nn.lr_scheduler import ParlAILRScheduler
from parlai.core.message import Message
from parlai.utils.distributed import is_distributed
from parlai.utils.misc import AttrDict, warn_once
from parlai.utils.fp16 import (
from parlai.core.metrics import (
from parlai.utils.distributed import is_primary_worker
from parlai.utils.torch import argsort, compute_grad_norm, padded_tensor, atomic_save
def _get_init_model(self, opt: Opt, shared):
    """
        Get model file to initialize with.

        If `init_model` exits, we will return the path to that file and maybe
        load dict file from that path. Otherwise, use `model_file.`

        :return:  path to load model from, whether we loaded from `init_model`
                  or not
        """
    init_model = None
    is_finetune = False
    if not shared:
        if opt.get('init_model') and os.path.isfile(opt['init_model']):
            init_model = opt['init_model']
            is_finetune = True
        if opt.get('model_file') and os.path.isfile(opt['model_file']):
            init_model = opt['model_file']
            is_finetune = False
        if opt.get('load_from_checkpoint') and opt.get('init_model') and opt['init_model'].endswith('.checkpoint'):
            init_model = opt['init_model']
            is_finetune = False
        if init_model is not None:
            if os.path.isfile(init_model + '.dict') or opt['dict_file'] is None:
                opt['dict_file'] = init_model + '.dict'
    return (init_model, is_finetune)