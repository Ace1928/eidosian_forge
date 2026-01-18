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
def _partial_hash(obj):
    h = hashlib.new('md5')
    h.update(_generate_hash(obj.args))
    h.update(_generate_hash(obj.func))
    h.update(_generate_hash(obj.keywords))
    return h.digest()