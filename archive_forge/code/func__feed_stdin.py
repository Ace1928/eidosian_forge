import collections
import subprocess
from . import events
from . import futures
from . import protocols
from . import streams
from . import tasks
from .coroutines import coroutine
from .log import logger
@coroutine
def _feed_stdin(self, input):
    debug = self._loop.get_debug()
    self.stdin.write(input)
    if debug:
        logger.debug('%r communicate: feed stdin (%s bytes)', self, len(input))
    try:
        yield from self.stdin.drain()
    except (BrokenPipeError, ConnectionResetError) as exc:
        if debug:
            logger.debug('%r communicate: stdin got %r', self, exc)
    if debug:
        logger.debug('%r communicate: close stdin', self)
    self.stdin.close()