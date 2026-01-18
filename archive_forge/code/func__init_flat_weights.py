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
def _init_flat_weights(self):
    self._flat_weights = [getattr(self, wn) if hasattr(self, wn) else None for wn in self._flat_weights_names]
    self._flat_weight_refs = [weakref.ref(w) if w is not None else None for w in self._flat_weights]
    self.flatten_parameters()