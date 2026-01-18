import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
def accepter(self):
    while True:
        try:
            c = self.listener.accept()
        except OSError:
            continue
        t = threading.Thread(target=self.handle_request, args=(c,))
        t.daemon = True
        t.start()