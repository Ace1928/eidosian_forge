import atexit
import contextlib
import functools
import inspect
import io
import os
import platform
import sys
import threading
import traceback
import debugpy
from debugpy.common import json, timestamp, util
class NoLog(object):
    file = filename = None
    __bool__ = __nonzero__ = lambda self: False

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass