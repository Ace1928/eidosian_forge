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
def _update_synced(self, event: param.parameterized.Event=None) -> None:
    if self._syncing:
        return
    query_params = self.query_params
    for p, parameters, _, on_error in self._synced:
        mapping = {v: k for k, v in parameters.items()}
        mapped = {}
        for k, v in query_params.items():
            if k not in mapping:
                continue
            pname = mapping[k]
            try:
                v = p.param[pname].deserialize(v)
            except Exception:
                pass
            try:
                equal = v == getattr(p, pname)
            except Exception:
                equal = False
            if not equal:
                mapped[pname] = v
        try:
            p.param.update(**mapped)
        except Exception:
            if on_error:
                on_error(mapped)