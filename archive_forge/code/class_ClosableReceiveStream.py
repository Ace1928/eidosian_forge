from __future__ import annotations
import os
import sys
from typing import TYPE_CHECKING
import trio
from .. import _core, _subprocess
from .._abc import ReceiveStream, SendStream  # noqa: TCH001
class ClosableReceiveStream(ReceiveStream):

    def close(self) -> None:
        ...