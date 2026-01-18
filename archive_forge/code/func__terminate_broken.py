import sys
import os
import threading
import collections
import time
import types
import weakref
import errno
from queue import Empty, Full
from . import connection
from . import context
from .util import debug, info, Finalize, register_after_fork, is_exiting
def _terminate_broken(self):
    self._reader.close()
    self.close()
    self.join_thread()