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
def add_child_handler(self, pid, callback, *args):
    loop = events.get_running_loop()
    thread = threading.Thread(target=self._do_waitpid, name=f'asyncio-waitpid-{next(self._pid_counter)}', args=(loop, pid, callback, args), daemon=True)
    self._threads[pid] = thread
    thread.start()