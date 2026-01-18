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
class XMLTestResultVerbose(_XMLTestResult):
    """
            Adding verbosity to test outputs:
            by default test summary prints 'skip',
            but we want to also print the skip reason.
            GH issue: https://github.com/pytorch/pytorch/issues/69014

            This works with unittest_xml_reporting<=3.2.0,>=2.0.0
            (3.2.0 is latest at the moment)
            """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        for c in self.callback.__closure__:
            if isinstance(c.cell_contents, str) and c.cell_contents == 'skip':
                c.cell_contents = f'skip: {reason}'

    def printErrors(self) -> None:
        super().printErrors()
        self.printErrorList('XPASS', self.unexpectedSuccesses)