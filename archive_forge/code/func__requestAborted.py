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
def _requestAborted(self, event):
    """
        Internal handler for when a request is aborted by a remote peer.

        @param event: The Hyper-h2 event that encodes information about the
            reset stream.
        @type event: L{h2.events.StreamReset}
        """
    stream = self.streams[event.stream_id]
    stream.connectionLost(Failure(ConnectionLost('Stream reset with code %s' % event.error_code)))
    self._requestDone(event.stream_id)