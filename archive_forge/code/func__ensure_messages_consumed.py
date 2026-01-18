import copy
import errno
import itertools
import os
import platform
import signal
import sys
import threading
import time
import warnings
from collections import deque
from functools import partial
from . import cpu_count, get_context
from . import util
from .common import (
from .compat import get_errno, mem_rss, send_offset
from .einfo import ExceptionInfo
from .dummy import DummyProcess
from .exceptions import (
from time import monotonic
from queue import Queue, Empty
from .util import Finalize, debug, warning
def _ensure_messages_consumed(self, completed):
    """ Returns true if all messages sent out have been received and
        consumed within a reasonable amount of time """
    if not self.on_ready_counter:
        return False
    for retry in range(GUARANTEE_MESSAGE_CONSUMPTION_RETRY_LIMIT):
        if self.on_ready_counter.value >= completed:
            debug('ensured messages consumed after %d retries', retry)
            return True
        time.sleep(GUARANTEE_MESSAGE_CONSUMPTION_RETRY_INTERVAL)
    warning('could not ensure all messages were consumed prior to exiting')
    return False