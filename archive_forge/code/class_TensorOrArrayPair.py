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
class TensorOrArrayPair(TensorLikePair):
    """Pair for tensor-like inputs.

    On the one hand this class is stricter than the builtin :class:`TensorLikePair` since it only allows instances of
    :class:`torch.Tensor` and :class:`numpy.ndarray` rather than allowing any tensor-like than can be converted into a
    tensor. On the other hand this class is looser since it converts all inputs into tensors with no regard of their
    relationship, e.g. comparing a :class:`torch.Tensor` to :class:`numpy.ndarray` is fine.

    In addition, this class supports overriding the absolute and relative tolerance through the ``@precisionOverride``
    and ``@toleranceOverride`` decorators.
    """

    def __init__(self, actual, expected, *, rtol_override=0.0, atol_override=0.0, **other_parameters):
        super().__init__(actual, expected, **other_parameters)
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    def _process_inputs(self, actual, expected, *, id, allow_subclasses):
        self._check_inputs_isinstance(actual, expected, cls=(torch.Tensor, np.ndarray))
        actual, expected = (self._to_tensor(input) for input in (actual, expected))
        for tensor in (actual, expected):
            self._check_supported(tensor, id=id)
        return (actual, expected)