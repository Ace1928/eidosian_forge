from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from bokeh.document import Document
from bokeh.server.contexts import BokehSessionContext, _RequestProxy
from bokeh.server.session import ServerSession
from bokeh.settings import settings
from bokeh.util.token import generate_jwt_token, generate_session_id
class ServerSessionStub(ServerSession):
    """
    Stubs out ServerSession methods since the session is only used for
    warming up the cache.
    """

    def _document_patched(self, event: DocumentPatchedEvent) -> None:
        return

    def _session_callback_added(self, event: SessionCallback):
        return

    def _session_callback_removed(self, event):
        return