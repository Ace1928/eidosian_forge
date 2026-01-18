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
class LazyTestBase(DeviceTypeTestBase):
    device_type = 'lazy'

    def _should_stop_test_suite(self):
        return False

    @classmethod
    def setUpClass(cls):
        import torch._lazy
        import torch._lazy.metrics
        import torch._lazy.ts_backend
        global lazy_ts_backend_init
        if not lazy_ts_backend_init:
            torch._lazy.ts_backend.init()
            lazy_ts_backend_init = True