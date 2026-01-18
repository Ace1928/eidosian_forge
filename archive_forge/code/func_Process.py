import collections
import itertools
import os
import queue
import threading
import time
import traceback
import types
import warnings
from . import util
from . import get_context, TimeoutError
from .connection import wait
@staticmethod
def Process(ctx, *args, **kwds):
    from .dummy import Process
    return Process(*args, **kwds)