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
def _generate_hash_inner(obj):
    hash_func = _find_hash_func(obj)
    if hash_func is not None:
        try:
            output = hash_func(obj)
        except BaseException as e:
            raise ValueError(f'User hash function {hash_func!r} failed for input {obj!r} with following error: {type(e).__name__}("{e}").') from e
        return output
    if hasattr(obj, '__reduce__'):
        h = hashlib.new('md5')
        try:
            reduce_data = obj.__reduce__()
        except BaseException:
            raise ValueError(f'Could not hash object of type {type(obj).__name__}') from None
        for item in reduce_data:
            h.update(_generate_hash(item))
        return h.digest()
    return _int_to_bytes(id(obj))