import torch
import unittest
from copy import deepcopy
from enum import Enum
from functools import wraps, partial
from itertools import chain, product
import itertools
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDNN
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_nn import nllloss_reference, get_reduction
from torch.testing._internal.common_utils import (
from types import ModuleType
from typing import List, Tuple, Type, Set, Dict
class ErrorModuleInput:
    """
    A ModuleInput that will cause the operation to throw an error plus information
    about the resulting error.
    """
    __slots__ = ['module_error_input', 'error_on', 'error_type', 'error_regex']

    def __init__(self, module_error_input, *, error_on=ModuleErrorEnum.CONSTRUCTION_ERROR, error_type=RuntimeError, error_regex):
        self.module_error_input = module_error_input
        self.error_on = error_on
        self.error_type = error_type
        self.error_regex = error_regex