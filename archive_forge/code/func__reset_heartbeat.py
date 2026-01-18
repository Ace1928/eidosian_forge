import asyncio
import base64
import binascii
import hashlib
import json
import sys
from typing import Any, Final, Iterable, Optional, Tuple, cast
import attr
from multidict import CIMultiDict
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import call_later, set_result
from .http import (
from .log import ws_logger
from .streams import EofStream, FlowControlDataQueue
from .typedefs import JSONDecoder, JSONEncoder
from .web_exceptions import HTTPBadRequest, HTTPException
from .web_request import BaseRequest
from .web_response import StreamResponse
def _reset_heartbeat(self) -> None:
    self._cancel_heartbeat()
    if self._heartbeat is not None:
        assert self._loop is not None
        self._heartbeat_cb = call_later(self._send_heartbeat, self._heartbeat, self._loop, timeout_ceil_threshold=self._req._protocol._timeout_ceil_threshold if self._req is not None else 5)