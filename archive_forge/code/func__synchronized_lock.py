import sys
from functools import partial
from inspect import isclass
from threading import Lock, RLock
from .arguments import formatargspec
from .__wrapt__ import (FunctionWrapper, BoundFunctionWrapper, ObjectProxy,
def _synchronized_lock(context):
    lock = vars(context).get('_synchronized_lock', None)
    if lock is None:
        with synchronized._synchronized_meta_lock:
            lock = vars(context).get('_synchronized_lock', None)
            if lock is None:
                lock = RLock()
                setattr(context, '_synchronized_lock', lock)
    return lock