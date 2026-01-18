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
def _cleanup_ttl(cache, ttl, time):
    """
    Deletes items in the cache if their TTL (time-to-live) has expired.
    """
    for key, (_, ts, _, _) in list(cache.items()):
        if time - ts > ttl:
            del cache[key]