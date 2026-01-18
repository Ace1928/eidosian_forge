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
def _bodyDataFinished_CONNECTED(self, reason=None):
    """
        Disconnect the protocol and move to the C{'FINISHED'} state.
        """
    if reason is None:
        reason = Failure(ResponseDone('Response body fully received'))
    self._bodyProtocol.connectionLost(reason)
    self._bodyProtocol = None
    self._state = 'FINISHED'