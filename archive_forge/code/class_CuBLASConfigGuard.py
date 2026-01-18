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
class CuBLASConfigGuard:
    cublas_var_name = 'CUBLAS_WORKSPACE_CONFIG'

    def __enter__(self):
        self.is_cuda10_2_or_higher = torch.version.cuda is not None and [int(x) for x in torch.version.cuda.split('.')] >= [10, 2]
        if self.is_cuda10_2_or_higher:
            self.cublas_config_restore = os.environ.get(self.cublas_var_name)
            os.environ[self.cublas_var_name] = ':4096:8'

    def __exit__(self, exception_type, exception_value, traceback):
        if self.is_cuda10_2_or_higher:
            cur_cublas_config = os.environ.get(self.cublas_var_name)
            if self.cublas_config_restore is None:
                if cur_cublas_config is not None:
                    del os.environ[self.cublas_var_name]
            else:
                os.environ[self.cublas_var_name] = self.cublas_config_restore