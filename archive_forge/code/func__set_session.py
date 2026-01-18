from __future__ import annotations
import logging # isort:skip
import weakref
from typing import (
from tornado import gen
from ..application.application import ServerContext, SessionContext
from ..document import Document
from ..protocol.exceptions import ProtocolError
from ..util.token import get_token_payload
from .session import ServerSession
def _set_session(self, session: ServerSession) -> None:
    self._session = session