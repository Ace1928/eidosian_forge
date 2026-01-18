import copy
import gc
import inspect
import runpy
import sys
import threading
from collections import namedtuple
from enum import Enum
from functools import wraps, partial
from typing import List, Any, ClassVar, Optional, Sequence, Tuple, Union, Dict, Set
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, \
from torch.testing._internal.common_cuda import _get_torch_cuda_version, \
from torch.testing._internal.common_dtype import get_all_dtypes
def _has_sufficient_memory(device, size):
    if torch.device(device).type == 'cuda':
        if not torch.cuda.is_available():
            return False
        gc.collect()
        torch.cuda.empty_cache()
        if device == 'cuda':
            device = 'cuda:0'
        return torch.cuda.memory.mem_get_info(device)[0] >= size
    if device == 'xla':
        raise unittest.SkipTest('TODO: Memory availability checks for XLA?')
    if device != 'cpu':
        raise unittest.SkipTest('Unknown device type')
    if not HAS_PSUTIL:
        raise unittest.SkipTest('Need psutil to determine if memory is sufficient')
    if TEST_WITH_ASAN or TEST_WITH_TSAN or TEST_WITH_UBSAN:
        effective_size = size * 10
    else:
        effective_size = size
    if psutil.virtual_memory().available < effective_size:
        gc.collect()
    return psutil.virtual_memory().available >= effective_size