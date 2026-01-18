import datetime as dt
import functools
import hashlib
import inspect
import io
import os
import pathlib
import pickle
import sys
import threading
import time
import unittest
import unittest.mock
import weakref
from contextlib import contextmanager
import param
from param.parameterized import iscoroutinefunction
from .state import state
def _numpy_hash(obj):
    h = hashlib.new('md5')
    h.update(_generate_hash(obj.shape))
    if obj.size >= _NP_SIZE_LARGE:
        import numpy as np
        state = np.random.RandomState(0)
        obj = state.choice(obj.flat, size=_NP_SAMPLE_SIZE)
    h.update(obj.tobytes())
    return h.digest()