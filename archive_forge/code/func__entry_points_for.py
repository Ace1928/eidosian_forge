import logging
import sys
from typing import Any, List
from .threading import run_once
from importlib.metadata import EntryPoint, entry_points
def _entry_points_for(key: str) -> List[EntryPoint]:
    if sys.version_info >= (3, 10) or _IMPORTLIB_META_VERSION >= (3, 6):
        return entry_points().select(group=key)
    else:
        return entry_points().get(key, {})