from __future__ import annotations
import io
import json
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _remove_prefix(self, input_string: str, suffix: str) -> str:
    if suffix and input_string.startswith(suffix):
        return input_string[len(suffix):]
    return input_string