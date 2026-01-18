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
def _assertNumberEqual(first, second, delta=None):
    if delta is None or first == second == 0.0 or math.isinf(first) or math.isinf(second):
        self.assertEqual(first, second, msg=msg)
        if not ignore_sign_on_zero:
            try:
                if math.copysign(1, first) != math.copysign(1, second):
                    self.fail(self._formatMessage(msg, '%s != %s' % (first, second)))
            except TypeError:
                pass
    else:
        self.assertAlmostEqual(first, second, delta=delta, msg=msg)