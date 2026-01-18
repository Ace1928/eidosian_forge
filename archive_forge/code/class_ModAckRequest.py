import typing
from typing import NamedTuple, Optional
class ModAckRequest(NamedTuple):
    ack_id: str
    seconds: float
    future: Optional['futures.Future']