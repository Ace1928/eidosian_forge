from __future__ import annotations
import copy
import json
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
@staticmethod
def _set_nested_dict(d: Dict, path: List[str], value: Any) -> None:
    """Set a value in a nested dictionary based on the given path."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value