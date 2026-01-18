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
class UnittestPair(Pair):
    """Fallback ABC pair that handles non-numeric inputs.

    To avoid recreating the mismatch messages of :meth:`unittest.TestCase.assertEqual`, this pair simply wraps it in
    order to use it with the :class:`Pair` "framework" from :func:`are_equal`.

    Define the :attr:`UnittestPair.CLS` in a subclass to indicate which class(es) of the inputs the pair should support.
    """
    CLS: Union[Type, Tuple[Type, ...]]
    TYPE_NAME: Optional[str] = None

    def __init__(self, actual, expected, **other_parameters):
        self._check_inputs_isinstance(actual, expected, cls=self.CLS)
        super().__init__(actual, expected, **other_parameters)

    def compare(self):
        test_case = unittest.TestCase()
        try:
            return test_case.assertEqual(self.actual, self.expected)
        except test_case.failureException as error:
            msg = str(error)
        type_name = self.TYPE_NAME or (self.CLS if isinstance(self.CLS, type) else self.CLS[0]).__name__
        self._fail(AssertionError, f'{type_name.title()} comparison failed: {msg}')