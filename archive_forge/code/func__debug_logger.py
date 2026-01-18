import collections
import contextlib
import itertools
import queue
import threading
import time
import memcache
from oslo_log import log
from oslo_cache._i18n import _
from oslo_cache import exception
def _debug_logger(self, msg, *args, **kwargs):
    self._do_log(log.DEBUG, msg, *args, **kwargs)