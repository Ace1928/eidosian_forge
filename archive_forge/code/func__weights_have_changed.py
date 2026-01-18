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
def _weights_have_changed(self):
    weights_changed = False
    for ref, name in zip(self._flat_weight_refs, self._flat_weights_names):
        weight = getattr(self, name) if hasattr(self, name) else None
        if weight is not None and ref is not None and (ref() is not weight):
            weights_changed = True
            break
    return weights_changed