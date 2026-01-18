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
def _set_code_close_transport(self, code: WSCloseCode) -> None:
    """Set the close code and close the transport."""
    self._close_code = code
    if self._req is not None and self._req.transport is not None:
        self._req.transport.close()