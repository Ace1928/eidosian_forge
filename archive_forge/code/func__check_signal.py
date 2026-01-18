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
def _check_signal(self, sig):
    """Internal helper to validate a signal.

        Raise ValueError if the signal number is invalid or uncatchable.
        Raise RuntimeError if there is a problem setting up the handler.
        """
    if not isinstance(sig, int):
        raise TypeError(f'sig must be an int, not {sig!r}')
    if sig not in signal.valid_signals():
        raise ValueError(f'invalid signal number {sig}')