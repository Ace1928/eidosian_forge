from __future__ import annotations
import json
import urllib.parse as urlparse
from typing import (
import param
from ..models.location import Location as _BkLocation
from ..reactive import Syncable
from ..util import edit_readonly, parse_query
from .document import create_doc_if_none_exists
from .state import state
def _server_destroy(self, session_context: BokehSessionContext) -> None:
    for p, ps, _, _ in self._synced:
        try:
            self.unsync(p, ps)
        except Exception:
            pass
    super()._server_destroy(session_context)