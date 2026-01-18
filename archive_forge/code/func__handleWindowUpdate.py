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
def _handleWindowUpdate(self, event):
    """
        Manage flow control windows.

        Streams that are blocked on flow control will register themselves with
        the connection. This will fire deferreds that wake those streams up and
        allow them to continue processing.

        @param event: The Hyper-h2 event that encodes information about the
            flow control window change.
        @type event: L{h2.events.WindowUpdated}
        """
    streamID = event.stream_id
    if streamID:
        if not self._streamIsActive(streamID):
            return
        if self._outboundStreamQueues.get(streamID):
            self.priority.unblock(streamID)
        self.streams[streamID].windowUpdated()
    else:
        for stream in self.streams.values():
            stream.windowUpdated()
            if self._outboundStreamQueues.get(stream.streamID):
                self.priority.unblock(stream.streamID)