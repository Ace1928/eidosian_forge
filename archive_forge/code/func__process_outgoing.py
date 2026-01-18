import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _process_outgoing(self):
    if not self._ssl_writing_paused:
        data = self._outgoing.read()
        if len(data):
            self._transport.write(data)
    self._control_app_writing()