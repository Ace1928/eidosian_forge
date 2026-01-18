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
class dtypes:

    def __init__(self, *args, device_type='all'):
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            for arg in args:
                assert isinstance(arg, (list, tuple)), f'When one dtype variant is a tuple or list, all dtype variants must be. Received non-list non-tuple dtype {str(arg)}'
                assert all((isinstance(dtype, torch.dtype) for dtype in arg)), f'Unknown dtype in {str(arg)}'
        else:
            assert all((isinstance(arg, torch.dtype) for arg in args)), f'Unknown dtype in {str(args)}'
        self.args = args
        self.device_type = device_type

    def __call__(self, fn):
        d = getattr(fn, 'dtypes', {})
        assert self.device_type not in d, f'dtypes redefinition for {self.device_type}'
        d[self.device_type] = self.args
        fn.dtypes = d
        return fn