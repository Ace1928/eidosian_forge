from __future__ import annotations
import asyncio
import http
import logging
from typing import Any, Callable, Literal, cast
from urllib.parse import unquote
import h11
from h11._connection import DEFAULT_MAX_INCOMPLETE_EVENT_SIZE
from uvicorn._types import (
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.protocols.http.flow_control import (
from uvicorn.protocols.utils import (
from uvicorn.server import ServerState
def _should_upgrade_to_ws(self) -> bool:
    if self.ws_protocol_class is None:
        if self.config.ws == 'auto':
            msg = 'Unsupported upgrade request.'
            self.logger.warning(msg)
            msg = 'No supported WebSocket library detected. Please use "pip install \'uvicorn[standard]\'", or install \'websockets\' or \'wsproto\' manually.'
            self.logger.warning(msg)
        return False
    return True