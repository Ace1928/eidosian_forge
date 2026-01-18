import time
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple
import zmq
from zmq import ENOTSOCK, ETERM, PUSH, QUEUE, Context, ZMQBindError, ZMQError, device
def _close_sockets(self):
    """Cleanup sockets we created"""
    for s in self._sockets:
        if s and (not s.closed):
            s.close()