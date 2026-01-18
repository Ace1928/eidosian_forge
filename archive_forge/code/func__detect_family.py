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
def _detect_family(self, numeric_object):
    """
        This function returns a string description of the type family
        that the object in question belongs to.  Possible return values
        are: "exact", "complex", "approximate", "sequence", and "unknown"
        """
    if isinstance(numeric_object, np.ndarray):
        return 'ndarray'
    if isinstance(numeric_object, enum.Enum):
        return 'enum'
    for tp in self._sequence_typesets:
        if isinstance(numeric_object, tp):
            return 'sequence'
    for tp in self._exact_typesets:
        if isinstance(numeric_object, tp):
            return 'exact'
    for tp in self._complex_types:
        if isinstance(numeric_object, tp):
            return 'complex'
    for tp in self._approx_typesets:
        if isinstance(numeric_object, tp):
            return 'approximate'
    return 'unknown'