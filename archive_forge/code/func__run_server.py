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
@classmethod
def _run_server(cls, registry, address, authkey, serializer, writer, initializer=None, initargs=()):
    """
        Create a server, report its address and run it
        """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if initializer is not None:
        initializer(*initargs)
    server = cls._Server(registry, address, authkey, serializer)
    writer.send(server.address)
    writer.close()
    util.info('manager serving at %r', server.address)
    server.serve_forever()