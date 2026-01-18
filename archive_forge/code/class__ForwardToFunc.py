from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
class _ForwardToFunc(SwiGLUOp):

    def __call__(self, *args, **kwargs):
        return self.op(*args, **kwargs)

    def info(self):
        if self.op.__name__ == 'no_such_operator':
            return 'not built'
        return 'available'