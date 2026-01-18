import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _control_ssl_reading(self):
    size = self._get_read_buffer_size()
    if size >= self._incoming_high_water and (not self._ssl_reading_paused):
        self._ssl_reading_paused = True
        self._transport.pause_reading()
    elif size <= self._incoming_low_water and self._ssl_reading_paused:
        self._ssl_reading_paused = False
        self._transport.resume_reading()