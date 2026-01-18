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
def _tryToWriteControlData(self):
    """
        Checks whether the connection is blocked on flow control and,
        if it isn't, writes any buffered control data.

        @return: L{True} if the connection is still active and
            L{False} if it was aborted because too many bytes have
            been written but not consumed by the other end.
        """
    bufferedBytes = self.conn.data_to_send()
    if not bufferedBytes:
        return True
    if self._consumerBlocked is None and (not self._bufferedControlFrames):
        self.transport.write(bufferedBytes)
        return True
    else:
        self._bufferedControlFrames.append(bufferedBytes)
        self._bufferedControlFrameBytes += len(bufferedBytes)
        if self._bufferedControlFrameBytes >= self._maxBufferedControlFrameBytes:
            maxBuffCtrlFrameBytes = self._maxBufferedControlFrameBytes
            self._log.error('Maximum number of control frame bytes buffered: {bufferedControlFrameBytes} > = {maxBufferedControlFrameBytes}. Aborting connection to client: {client} ', bufferedControlFrameBytes=self._bufferedControlFrameBytes, maxBufferedControlFrameBytes=maxBuffCtrlFrameBytes, client=self.transport.getPeer())
            self.transport.abortConnection()
            self.connectionLost(Failure(ExcessiveBufferingError()))
            return False
        return True