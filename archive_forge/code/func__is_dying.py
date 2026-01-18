import atexit
import queue
import threading
import weakref
def _is_dying(self):
    if self.should_stop or _dying:
        return True
    executor = self.executor_ref()
    if executor is None:
        return True
    del executor
    return False