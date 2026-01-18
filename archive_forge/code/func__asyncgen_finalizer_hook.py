import collections
import collections.abc
import concurrent.futures
import errno
import functools
import heapq
import itertools
import os
import socket
import stat
import subprocess
import threading
import time
import traceback
import sys
import warnings
import weakref
from . import constants
from . import coroutines
from . import events
from . import exceptions
from . import futures
from . import protocols
from . import sslproto
from . import staggered
from . import tasks
from . import transports
from . import trsock
from .log import logger
def _asyncgen_finalizer_hook(self, agen):
    self._asyncgens.discard(agen)
    if not self.is_closed():
        self.call_soon_threadsafe(self.create_task, agen.aclose())