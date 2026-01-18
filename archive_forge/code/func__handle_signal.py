import errno
import io
import itertools
import os
import selectors
import signal
import socket
import stat
import subprocess
import sys
import threading
import warnings
from . import base_events
from . import base_subprocess
from . import constants
from . import coroutines
from . import events
from . import exceptions
from . import futures
from . import selector_events
from . import tasks
from . import transports
from .log import logger
def _handle_signal(self, sig):
    """Internal helper that is the actual signal handler."""
    handle = self._signal_handlers.get(sig)
    if handle is None:
        return
    if handle._cancelled:
        self.remove_signal_handler(sig)
    else:
        self._add_callback_signalsafe(handle)