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
def _project_vec(self, vec, target_dim, method='random'):
    """
        If needed, project vector to target dimensionality.

        Projection methods implemented are the following:

        random - random gaussian matrix multiplication of input vector

        :param vec:
            one-dimensional vector

        :param target_dim:
            dimension of returned vector

        :param method:
            projection method. will be used even if the dim is not changing if
            method ends in "-force".
        """
    pre_dim = vec.size(0)
    if pre_dim != target_dim or method.endswith('force'):
        if method.startswith('random'):
            if not hasattr(self, 'proj_rp'):
                self.proj_rp = torch.Tensor(pre_dim, target_dim).normal_()
                self.proj_rp /= target_dim
            return torch.mm(vec.unsqueeze(0), self.proj_rp)
        else:
            raise RuntimeError('Projection method not implemented: {}'.format(method))
    else:
        return vec