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
def _update_param_kwargs(param_kwargs, name, value):
    """ Adds a kwarg with the specified name and value to the param_kwargs dict. """
    plural_name = f'{name}s'
    if name in param_kwargs:
        del param_kwargs[name]
    if plural_name in param_kwargs:
        del param_kwargs[plural_name]
    if isinstance(value, (list, tuple)):
        param_kwargs[plural_name] = value
    elif value is not None:
        param_kwargs[name] = value