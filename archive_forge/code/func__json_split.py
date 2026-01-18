from __future__ import annotations
import copy
import json
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
def _json_split(self, data: Dict[str, Any], current_path: List[str]=[], chunks: List[Dict]=[{}]) -> List[Dict]:
    """
        Split json into maximum size dictionaries while preserving structure.
        """
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = current_path + [key]
            chunk_size = self._json_size(chunks[-1])
            size = self._json_size({key: value})
            remaining = self.max_chunk_size - chunk_size
            if size < remaining:
                self._set_nested_dict(chunks[-1], new_path, value)
            else:
                if chunk_size >= self.min_chunk_size:
                    chunks.append({})
                self._json_split(value, new_path, chunks)
    else:
        self._set_nested_dict(chunks[-1], current_path, data)
    return chunks