import atexit
import struct
import warnings
from collections import namedtuple
from os import getpid
from threading import Event, Lock, Thread
import zmq
class GarbageCollectorThread(Thread):
    """Thread in which garbage collection actually happens."""

    def __init__(self, gc):
        super().__init__()
        self.gc = gc
        self.daemon = True
        self.pid = getpid()
        self.ready = Event()

    def run(self):
        if getpid is None or getpid() != self.pid:
            self.ready.set()
            return
        try:
            s = self.gc.context.socket(zmq.PULL)
            s.linger = 0
            s.bind(self.gc.url)
        finally:
            self.ready.set()
        while True:
            if getpid is None or getpid() != self.pid:
                return
            msg = s.recv()
            if msg == b'DIE':
                break
            fmt = 'L' if len(msg) == 4 else 'Q'
            key = struct.unpack(fmt, msg)[0]
            tup = self.gc.refs.pop(key, None)
            if tup and tup.event:
                tup.event.set()
            del tup
        s.close()