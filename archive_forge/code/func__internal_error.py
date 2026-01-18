from __future__ import annotations
import logging # isort:skip
import calendar
import datetime as dt
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse
from tornado import locks, web
from tornado.websocket import WebSocketClosedError, WebSocketHandler
from bokeh.settings import settings
from bokeh.util.token import check_token_signature, get_session_id, get_token_payload
from ...protocol import Protocol
from ...protocol.exceptions import MessageError, ProtocolError, ValidationError
from ...protocol.message import Message
from ...protocol.receiver import Receiver
from ...util.dataclasses import dataclass
from ..protocol_handler import ProtocolHandler
from .auth_request_handler import AuthRequestHandler
def _internal_error(self, message: str) -> None:
    log.error('Bokeh Server internal error: %s, closing connection', message)
    self.close(10000, message)