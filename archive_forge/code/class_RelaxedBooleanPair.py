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
class RelaxedBooleanPair(BooleanPair):
    """Pair for boolean-like inputs.

    In contrast to the builtin :class:`BooleanPair`, this class also supports one input being a number or a single
    element tensor-like.
    """
    _supported_number_types = NumberPair(0, 0)._supported_types

    def _process_inputs(self, actual, expected, *, id):
        tensor_or_array_types: Tuple[Type, ...] = (torch.Tensor, np.ndarray)
        other_supported_types = (*self._supported_types, *self._supported_number_types, *tensor_or_array_types)
        if not (isinstance(actual, self._supported_types) and isinstance(expected, other_supported_types) or (isinstance(expected, self._supported_types) and isinstance(actual, other_supported_types))):
            self._inputs_not_supported()
        return [self._to_bool(input, id=id) for input in (actual, expected)]

    def _to_bool(self, bool_like, *, id):
        if isinstance(bool_like, np.number):
            return bool(bool_like.item())
        elif type(bool_like) in self._supported_number_types:
            return bool(bool_like)
        elif isinstance(bool_like, (torch.Tensor, np.ndarray)):
            numel = bool_like.numel() if isinstance(bool_like, torch.Tensor) else bool_like.size
            if numel > 1:
                self._fail(ValueError, f'Only single element tensor-likes can be compared against a boolean. Got {numel} elements instead.', id=id)
            return bool(bool_like.item())
        else:
            return super()._to_bool(bool_like, id=id)