from __future__ import annotations
import asyncio
import codecs
import collections
import logging
import random
import ssl
import struct
import sys
import time
import uuid
import warnings
from typing import (
from ..datastructures import Headers
from ..exceptions import (
from ..extensions import Extension
from ..frames import (
from ..protocol import State
from ..typing import Data, LoggerLike, Subprotocol
from .compatibility import asyncio_timeout
from .framing import Frame
def abort_pings(self) -> None:
    """
        Raise ConnectionClosed in pending keepalive pings.

        They'll never receive a pong once the connection is closed.

        """
    assert self.state is State.CLOSED
    exc = self.connection_closed_exc()
    for pong_waiter, _ping_timestamp in self.pings.values():
        pong_waiter.set_exception(exc)
        pong_waiter.cancel()