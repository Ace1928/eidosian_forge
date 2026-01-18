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
@classmethod
def _get_dtypes(cls, test):
    if not hasattr(test, 'dtypes'):
        return None
    default_dtypes = test.dtypes.get('all')
    msg = f"@dtypes is mandatory when using @dtypesIf however '{test.__name__}' didn't specify it"
    assert default_dtypes is not None, msg
    return test.dtypes.get(cls.device_type, default_dtypes)