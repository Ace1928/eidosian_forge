from _pydevd_bundle.pydevd_constants import (STATE_RUN, PYTHON_SUSPEND, SUPPORT_GEVENT, ForkSafeLock,
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import threading
import weakref
def _is_stepping(self):
    if self.pydev_state == STATE_RUN and self.pydev_step_cmd != -1:
        return True
    if self.pydev_state == STATE_SUSPEND and self.is_in_wait_loop:
        return True
    return False