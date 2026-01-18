from __future__ import annotations
import struct
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Sequence, Tuple
from .. import extensions, frames
from ..exceptions import PayloadTooBig, ProtocolError
from ..frames import (  # noqa: E402, F401, I001

        Write a WebSocket frame.

        Args:
            frame: Frame to write.
            write: Function that writes bytes.
            mask: Whether the frame should be masked i.e. whether the write
                happens on the client side.
            extensions: List of extensions, applied in order.

        Raises:
            ProtocolError: If the frame contains incorrect values.

        