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
def _pandas_hash(obj):
    import pandas as pd
    if not isinstance(obj, (pd.Series, pd.DataFrame)):
        obj = pd.Series(obj)
    if len(obj) >= _PANDAS_ROWS_LARGE:
        obj = obj.sample(n=_PANDAS_SAMPLE_SIZE, random_state=0)
    try:
        return b'%s' % pd.util.hash_pandas_object(obj).sum()
    except TypeError:
        return b'%s' % pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)