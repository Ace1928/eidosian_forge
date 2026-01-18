from typing import NamedTuple
class FlowControlSettings(NamedTuple):
    messages_outstanding: int
    bytes_outstanding: int