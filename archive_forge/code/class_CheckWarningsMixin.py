import cmath
import contextlib
from collections import defaultdict
import enum
import gc
import math
import platform
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import io
import ctypes
import multiprocessing as mp
import warnings
import traceback
from contextlib import contextmanager
import uuid
import importlib
import types as pytypes
from functools import cached_property
import numpy as np
from numba import testing, types
from numba.core import errors, typing, utils, config, cpu
from numba.core.typing import cffi_utils
from numba.core.compiler import (compile_extra, Flags,
from numba.core.typed_passes import IRLegalization
from numba.core.untyped_passes import PreserveIR
import unittest
from numba.core.runtime import rtsys
from numba.np import numpy_support
from numba.core.runtime import _nrt_python as _nrt
from numba.core.extending import (
from numba.core.datamodel.models import OpaqueModel
class CheckWarningsMixin(object):

    @contextlib.contextmanager
    def check_warnings(self, messages, category=RuntimeWarning):
        with warnings.catch_warnings(record=True) as catch:
            warnings.simplefilter('always')
            yield
        found = 0
        for w in catch:
            for m in messages:
                if m in str(w.message):
                    self.assertEqual(w.category, category)
                    found += 1
        self.assertEqual(found, len(messages))