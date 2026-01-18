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
def _deliverBody_DEFERRED_CLOSE(self, protocol):
    """
        Deliver any buffered data to C{protocol} and then disconnect the
        protocol.  Move to the C{'FINISHED'} state.
        """
    protocol.makeConnection(self._transport)
    for data in self._bodyBuffer:
        protocol.dataReceived(data)
    self._bodyBuffer = None
    protocol.connectionLost(self._reason)
    self._state = 'FINISHED'