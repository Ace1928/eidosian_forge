import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _get_app_transport(self):
    if self._app_transport is None:
        if self._app_transport_created:
            raise RuntimeError('Creating _SSLProtocolTransport twice')
        self._app_transport = _SSLProtocolTransport(self._loop, self)
        self._app_transport_created = True
    return self._app_transport