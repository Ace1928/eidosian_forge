from __future__ import annotations
import copy
import json
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
def _list_to_dict_preprocessing(self, data: Any) -> Any:
    if isinstance(data, dict):
        return {k: self._list_to_dict_preprocessing(v) for k, v in data.items()}
    elif isinstance(data, list):
        return {str(i): self._list_to_dict_preprocessing(item) for i, item in enumerate(data)}
    else:
        return data