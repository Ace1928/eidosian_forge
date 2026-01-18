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
class TrackedInputIter:

    def __init__(self, child_iter, input_type_desc, callback=lambda x: x):
        self.child_iter = enumerate(child_iter)
        self.input_type_desc = input_type_desc
        self.callback = callback
        self.test_fn = extract_test_fn()

    def __iter__(self):
        return self

    def __next__(self):
        input_idx, input_val = next(self.child_iter)
        self._set_tracked_input(TrackedInput(index=input_idx, val=self.callback(input_val), type_desc=self.input_type_desc))
        return input_val

    def _set_tracked_input(self, tracked_input: TrackedInput):
        if self.test_fn is None:
            return
        if not hasattr(self.test_fn, 'tracked_input'):
            return
        self.test_fn.tracked_input = tracked_input