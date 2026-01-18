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
def accept_connection(self, c, name):
    """
        Spawn a new thread to serve this connection
        """
    threading.current_thread().name = name
    c.send(('#RETURN', None))
    self.serve_client(c)