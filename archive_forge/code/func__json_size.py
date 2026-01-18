from __future__ import annotations
import copy
import json
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
@staticmethod
def _json_size(data: Dict) -> int:
    """Calculate the size of the serialized JSON object."""
    return len(json.dumps(data))