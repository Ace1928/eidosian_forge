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
def _get_embtype(self, emb_type):
    if emb_type.startswith('glove'):
        init = 'glove'
        from parlai.zoo.glove_vectors.build import download
        embs = download(self.opt.get('datapath'))
    elif emb_type.startswith('fasttext_cc'):
        init = 'fasttext_cc'
        from parlai.zoo.fasttext_cc_vectors.build import download
        embs = download(self.opt.get('datapath'))
    elif emb_type.startswith('fasttext'):
        init = 'fasttext'
        from parlai.zoo.fasttext_vectors.build import download
        embs = download(self.opt.get('datapath'))
    else:
        raise RuntimeError('embedding type {} not implemented. check arg, submit PR to this function, or override it.'.format(emb_type))
    return (embs, init)