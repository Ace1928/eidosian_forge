import bisect
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
def _InitializeLoader(self):
    path = self._GetNextPath()
    if path:
        self._SetPath(path)