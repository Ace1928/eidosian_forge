import argparse
import contextlib
import copy
import ctypes
import errno
import functools
import gc
import inspect
import io
import json
import logging
import math
import operator
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from itertools import product, chain
from pathlib import Path
from statistics import mean
from typing import (
from unittest.mock import MagicMock
import expecttest
import numpy as np
import __main__  # type: ignore[import]
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mps
import torch.backends.xnnpack
import torch.cuda
from torch import Tensor
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
from torch._utils_internal import get_writable_path
from torch.nn import (
from torch.onnx import (
from torch.testing import make_tensor
from torch.testing._comparison import (
from torch.testing._comparison import not_close_error_metas
from torch.testing._internal.common_dtype import get_all_dtypes
import torch.utils._pytree as pytree
from .composite_compliance import no_dispatch
class decorateIf(_TestParametrizer):
    """
    Decorator for applying parameter-specific conditional decoration.
    Composes with other test parametrizers (e.g. @modules, @ops, @parametrize, etc.).

    Examples::

        @decorateIf(unittest.skip, lambda params: params["x"] == 2)
        @parametrize("x", range(5))
        def test_foo(self, x):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')])
        @decorateIf(
            unittest.expectedFailure,
            lambda params: params["x"] == 3 and params["y"] == "baz"
        )
        def test_bar(self, x, y):
            ...

        @decorateIf(
            unittest.expectedFailure,
            lambda params: params["op"].name == "add" and params["dtype"] == torch.float16
        )
        @ops(op_db)
        def test_op_foo(self, device, dtype, op):
            ...

        @decorateIf(
            unittest.skip,
            lambda params: params["module_info"].module_cls is torch.nn.Linear and                 params["device"] == "cpu"
        )
        @modules(module_db)
        def test_module_foo(self, device, dtype, module_info):
            ...

    Args:
        decorator: Test decorator to apply if the predicate is satisfied.
        predicate_fn (Callable): Function taking in a dict of params and returning a boolean
            indicating whether the decorator should be applied or not.
    """

    def __init__(self, decorator, predicate_fn):
        self.decorator = decorator
        self.predicate_fn = predicate_fn

    def _parametrize_test(self, test, generic_cls, device_cls):

        def decorator_fn(params, decorator=self.decorator, predicate_fn=self.predicate_fn):
            if predicate_fn(params):
                return [decorator]
            else:
                return []

        @wraps(test)
        def test_wrapper(*args, **kwargs):
            return test(*args, **kwargs)
        test_name = ''
        yield (test_wrapper, test_name, {}, decorator_fn)