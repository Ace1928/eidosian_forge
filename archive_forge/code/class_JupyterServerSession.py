from __future__ import annotations
import asyncio
import html
import os
import weakref
from dataclasses import dataclass
from typing import (
import tornado
from bokeh.document import Document
from bokeh.embed.bundle import extension_dirs
from bokeh.protocol import Protocol
from bokeh.protocol.receiver import Receiver
from bokeh.server.connection import ServerConnection
from bokeh.server.contexts import BokehSessionContext
from bokeh.server.protocol_handler import ProtocolHandler
from bokeh.server.session import ServerSession
from bokeh.server.views.static_handler import StaticHandler
from bokeh.server.views.ws import WSHandler
from bokeh.util.token import get_session_id, get_token_payload
from ipykernel.comm import Comm
from ..util import edit_readonly
from .application import build_single_handler_application
from .resources import Resources
from .server import server_html_page_for_session
from .state import set_curdoc, state
class JupyterServerSession(ServerSession):
    _tasks = set()

    def _document_patched(self, event: DocumentPatchedEvent) -> None:
        may_suppress = event.setter is self
        for connection in self._subscribed_connections:
            if may_suppress and connection is self._current_patch_connection:
                continue
            task = asyncio.ensure_future(connection.send_patch_document(event))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)