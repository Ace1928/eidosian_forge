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
def _sig_chld(self, signum, frame):
    try:
        self._do_waitpid_all()
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:
        logger.warning('Unknown exception in SIGCHLD handler', exc_info=True)