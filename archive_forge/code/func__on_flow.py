import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def _on_flow(self, active):
    """Enable/disable flow from peer.

        This method asks the peer to pause or restart the flow of
        content data. This is a simple flow-control mechanism that a
        peer can use to avoid overflowing its queues or otherwise
        finding itself receiving more messages than it can process.
        Note that this method is not intended for window control.  The
        peer that receives a request to stop sending content should
        finish sending the current content, if any, and then wait
        until it receives a Flow restart method.

        RULE:

            When a new channel is opened, it is active.  Some
            applications assume that channels are inactive until
            started.  To emulate this behaviour a client MAY open the
            channel, then pause it.

        RULE:

            When sending content data in multiple frames, a peer
            SHOULD monitor the channel for incoming methods and
            respond to a Channel.Flow as rapidly as possible.

        RULE:

            A peer MAY use the Channel.Flow method to throttle
            incoming content data for internal reasons, for example,
            when exchanging data over a slower connection.

        RULE:

            The peer that requests a Channel.Flow method MAY
            disconnect and/or ban a peer that does not respect the
            request.

        PARAMETERS:
            active: boolean

                start/stop content frames

                If True, the peer starts sending content frames.  If
                False, the peer stops sending content frames.
        """
    self.active = active
    self._x_flow_ok(self.active)