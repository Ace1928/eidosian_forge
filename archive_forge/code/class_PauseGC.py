import gc
from pyomo.common.multithread import MultiThreadWrapper
class PauseGC(object):
    __slots__ = ('reenable_gc', 'stack_pointer')

    def __init__(self):
        self.stack_pointer = None
        self.reenable_gc = None

    def __enter__(self):
        if self.stack_pointer:
            raise RuntimeError('Entering PauseGC context manager that was already entered.')
        PauseGCCompanion._stack_depth += 1
        self.stack_pointer = PauseGCCompanion._stack_depth
        self.reenable_gc = gc.isenabled()
        if self.reenable_gc:
            gc.disable()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if not self.stack_pointer:
            return
        if PauseGCCompanion._stack_depth:
            if PauseGCCompanion._stack_depth != self.stack_pointer:
                raise RuntimeError('Exiting PauseGC context manager out of order: there are other active PauseGC context managers that were entered after this context manager and have not yet been exited.')
            PauseGCCompanion._stack_depth -= 1
            self.stack_pointer = None
        if self.reenable_gc:
            gc.enable()
        self.reenable_gc = None