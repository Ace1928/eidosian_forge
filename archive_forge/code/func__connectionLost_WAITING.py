import re
from zope.interface import implementer
from twisted.internet.defer import (
from twisted.internet.error import ConnectionDone
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.reflect import fullyQualifiedName
from twisted.web.http import (
from twisted.web.http_headers import Headers
from twisted.web.iweb import UNKNOWN_LENGTH, IClientRequest, IResponse
def _connectionLost_WAITING(self, reason):
    """
        Disconnect the response parser so that it can propagate the event as
        necessary (for example, to call an application protocol's
        C{connectionLost} method, or to fail a request L{Deferred}) and move
        to the C{'CONNECTION_LOST'} state.
        """
    self._disconnectParser(reason)
    self._state = 'CONNECTION_LOST'