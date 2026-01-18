import io
from collections import deque
from typing import List
from zope.interface import implementer
import h2.config
import h2.connection
import h2.errors
import h2.events
import h2.exceptions
import priority
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.policies import TimeoutMixin
from twisted.python.failure import Failure
from twisted.web.error import ExcessiveBufferingError
def _requestReceived(self, event):
    """
        Internal handler for when a request has been received.

        @param event: The Hyper-h2 event that encodes information about the
            received request.
        @type event: L{h2.events.RequestReceived}
        """
    stream = H2Stream(event.stream_id, self, event.headers, self.requestFactory, self.site, self.factory)
    self.streams[event.stream_id] = stream
    self._streamCleanupCallbacks[event.stream_id] = Deferred()
    self._outboundStreamQueues[event.stream_id] = deque()
    try:
        self.priority.insert_stream(event.stream_id)
    except priority.DuplicateStreamError:
        pass
    else:
        self.priority.block(event.stream_id)