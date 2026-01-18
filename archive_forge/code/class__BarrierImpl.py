import enum
import threading
from cupyx.distributed import _klv_utils
class _BarrierImpl:

    def __init__(self, world_size):
        self._world_size = world_size
        self._cvar = threading.Condition()

    def __call__(self):
        with self._cvar:
            self._world_size -= 1
            if self._world_size == 0:
                self._cvar.notifyAll()
            elif self._world_size > 0:
                self._cvar.wait()