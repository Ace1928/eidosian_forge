import typing
from typing import NamedTuple, Optional
class DropRequest(NamedTuple):
    ack_id: str
    byte_size: int
    ordering_key: Optional[str]