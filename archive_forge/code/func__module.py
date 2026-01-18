from contextlib import contextmanager
import dbm
import os
import threading
from ..api import BytesBackend
from ..api import NO_VALUE
from ... import util
@util.memoized_property
def _module(self):
    import fcntl
    return fcntl