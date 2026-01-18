from typing import cast, List, Type, Union, ValuesView
from .._connection import Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import CLIENT, CLOSED, DONE, MUST_CLOSE, SERVER
from .._util import Sentinel
class ConnectionPair:

    def __init__(self) -> None:
        self.conn = {CLIENT: Connection(CLIENT), SERVER: Connection(SERVER)}
        self.other = {CLIENT: SERVER, SERVER: CLIENT}

    @property
    def conns(self) -> ValuesView[Connection]:
        return self.conn.values()

    def send(self, role: Type[Sentinel], send_events: Union[List[Event], Event], expect: Union[List[Event], Event, Literal['match']]='match') -> bytes:
        if not isinstance(send_events, list):
            send_events = [send_events]
        data = b''
        closed = False
        for send_event in send_events:
            new_data = self.conn[role].send(send_event)
            if new_data is None:
                closed = True
            else:
                data += new_data
        if data:
            self.conn[self.other[role]].receive_data(data)
        if closed:
            self.conn[self.other[role]].receive_data(b'')
        got_events = get_all_events(self.conn[self.other[role]])
        if expect == 'match':
            expect = send_events
        if not isinstance(expect, list):
            expect = [expect]
        assert got_events == expect
        return data