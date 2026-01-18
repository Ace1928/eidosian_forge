from __future__ import annotations
import pickle
from io import BytesIO
from typing import (
from rdflib.events import Dispatcher, Event
def _get_ids(self, key: Any) -> Optional[str]:
    try:
        return self._ids.get(key)
    except TypeError:
        return None