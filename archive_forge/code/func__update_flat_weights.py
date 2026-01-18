import math
import warnings
import numbers
import weakref
from typing import List, Tuple, Optional, overload
import torch
from torch import Tensor
from .module import Module
from ..parameter import Parameter
from ..utils.rnn import PackedSequence
from .. import init
from ... import _VF
def _update_flat_weights(self):
    if not torch.jit.is_scripting():
        if self._weights_have_changed():
            self._init_flat_weights()