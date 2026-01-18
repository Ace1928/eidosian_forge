import typing
from typing_extensions import Protocol
class SizedStringProtocol(Protocol, StringProtocol, typing.Sized):
    pass