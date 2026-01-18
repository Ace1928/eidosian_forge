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
def _cancel_heartbeat(self) -> None:
    if self._pong_response_cb is not None:
        self._pong_response_cb.cancel()
        self._pong_response_cb = None
    if self._heartbeat_cb is not None:
        self._heartbeat_cb.cancel()
        self._heartbeat_cb = None