import typing
from typing_extensions import Protocol
class StringProtocol(Protocol):

    def __str__(self) -> str:
        ...