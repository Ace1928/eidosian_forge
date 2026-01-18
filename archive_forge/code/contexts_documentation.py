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
 Server-side holder for ``bokeh.application.Application`` plus any associated data.
        This holds data that's global to all sessions, while ``ServerSession`` holds
        data specific to an "instance" of the application.

    