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
class precisionOverride:

    def __init__(self, d):
        assert isinstance(d, dict), 'precisionOverride not given a dtype : precision dict!'
        for dtype in d.keys():
            assert isinstance(dtype, torch.dtype), f'precisionOverride given unknown dtype {dtype}'
        self.d = d

    def __call__(self, fn):
        fn.precision_overrides = self.d
        return fn