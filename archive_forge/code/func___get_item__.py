import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def __get_item__(self, idx):
    """
        ensures compatibility with older quant state scheme with nested lists.
        assumes the following layout:
        state = [qabsmax, input_shape, A.dtype, blocksize, [offset, state2], quant_type]
        state2 = [absmax, input_shape, A.dtype, blocksize, None, quant_type]
        """
    if self.nested:
        list_repr = [self.absmax, self.shape, self.dtype, self.blocksize, [self.offset, self.state2], self.quant_type]
    else:
        list_repr = [self.absmax, self.shape, self.dtype, self.blocksize, None, self.quant_type]
    return list_repr[idx]