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
def _handlePriorityUpdate(self, event):
    """
        Internal handler for when a stream priority is updated.

        @param event: The Hyper-h2 event that encodes information about the
            stream reprioritization.
        @type event: L{h2.events.PriorityUpdated}
        """
    try:
        self.priority.reprioritize(stream_id=event.stream_id, depends_on=event.depends_on or None, weight=event.weight, exclusive=event.exclusive)
    except priority.MissingStreamError:
        self.priority.insert_stream(stream_id=event.stream_id, depends_on=event.depends_on or None, weight=event.weight, exclusive=event.exclusive)
        self.priority.block(event.stream_id)