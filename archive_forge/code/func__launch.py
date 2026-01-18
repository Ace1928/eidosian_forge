import os
import signal
from . import util
def _launch(self, process_obj):
    code = 1
    parent_r, child_w = os.pipe()
    child_r, parent_w = os.pipe()
    self.pid = os.fork()
    if self.pid == 0:
        try:
            os.close(parent_r)
            os.close(parent_w)
            code = process_obj._bootstrap(parent_sentinel=child_r)
        finally:
            os._exit(code)
    else:
        os.close(child_w)
        os.close(child_r)
        self.finalizer = util.Finalize(self, util.close_fds, (parent_r, parent_w))
        self.sentinel = parent_r