import subprocess
from . import events
from . import protocols
from . import streams
from . import tasks
from .log import logger
def _maybe_close_transport(self):
    if len(self._pipe_fds) == 0 and self._process_exited:
        self._transport.close()
        self._transport = None