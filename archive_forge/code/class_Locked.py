import threading
import fasteners
from fasteners import test
class Locked(object):

    def __init__(self):
        self._lock = threading.Lock()

    @fasteners.locked
    def i_am_locked(self, cb):
        cb(self._lock.locked())

    def i_am_not_locked(self, cb):
        cb(self._lock.locked())