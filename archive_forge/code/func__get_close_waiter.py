import subprocess
from . import events
from . import protocols
from . import streams
from . import tasks
from .log import logger
def _get_close_waiter(self, stream):
    if stream is self.stdin:
        return self._stdin_closed