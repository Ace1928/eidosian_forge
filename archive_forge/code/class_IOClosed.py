import importlib
import inspect
import os
import warnings
from eventlet import patcher
from eventlet.support import greenlets as greenlet
from eventlet import timeout
class IOClosed(IOError):
    pass