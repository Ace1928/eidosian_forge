from __future__ import annotations
import sys
import types
from typing import (
class ASGI2Protocol(Protocol):

    def __init__(self, scope: Scope) -> None:
        ...

    async def __call__(self, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        ...