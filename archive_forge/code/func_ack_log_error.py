from __future__ import annotations
import sys
from .compression import decompress
from .exceptions import MessageStateError, reraise
from .serialization import loads
from .utils.functional import dictfilter
def ack_log_error(self, logger, errors, multiple=False):
    try:
        self.ack(multiple=multiple)
    except BrokenPipeError as exc:
        logger.critical("Couldn't ack %r, reason:%r", self.delivery_tag, exc, exc_info=True)
        raise
    except errors as exc:
        logger.critical("Couldn't ack %r, reason:%r", self.delivery_tag, exc, exc_info=True)