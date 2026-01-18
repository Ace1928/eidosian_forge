from abc import ABC
from dataclasses import dataclass, field
from typing import Generic, List, Optional, Sequence, TypeVar, Union
from .extensions import Extension
from .typing import Headers
@dataclass(frozen=True)
class CloseConnection(Event):
    """The end of a Websocket connection, represents a closure frame.

    **wsproto does not automatically send a response to a close event.** To
    comply with the RFC you MUST send a close event back to the remote WebSocket
    if you have not already sent one. The :meth:`response` method provides a
    suitable event for this purpose, and you should check if a response needs
    to be sent by checking :func:`wsproto.WSConnection.state`.

    Fields:

    .. attribute:: code

       (Required) The integer close code to indicate why the connection
       has closed.

    .. attribute:: reason

       Additional reasoning for why the connection has closed.

    """
    code: int
    reason: Optional[str] = None

    def response(self) -> 'CloseConnection':
        """Generate an RFC-compliant close frame to send back to the peer."""
        return CloseConnection(code=self.code, reason=self.reason)