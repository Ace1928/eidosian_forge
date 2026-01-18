import atexit
import struct
import warnings
from collections import namedtuple
from os import getpid
from threading import Event, Lock, Thread
import zmq
@property
def _push_socket(self):
    """The PUSH socket for use in the zmq message destructor callback."""
    if getattr(self, '_stay_down', False):
        raise RuntimeError('zmq gc socket requested during shutdown')
    if not self.is_alive() or self._push is None:
        self._push = self.context.socket(zmq.PUSH)
        self._push.connect(self.url)
    return self._push