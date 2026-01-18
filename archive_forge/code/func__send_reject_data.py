from collections import deque
from typing import (
import h11
from .connection import Connection, ConnectionState, ConnectionType
from .events import AcceptConnection, Event, RejectConnection, RejectData, Request
from .extensions import Extension
from .typing import Headers
from .utilities import (
def _send_reject_data(self, event: RejectData) -> bytes:
    if self.state != ConnectionState.REJECTING:
        raise LocalProtocolError(f'Cannot send rejection data in state {self.state}')
    data = self._h11_connection.send(h11.Data(data=event.data)) or b''
    if event.body_finished:
        data += self._h11_connection.send(h11.EndOfMessage()) or b''
        self._state = ConnectionState.CLOSED
    return data