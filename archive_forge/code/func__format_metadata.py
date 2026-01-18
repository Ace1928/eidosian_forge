from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _format_metadata(self, row_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Helper method to format the metadata from the Couchbase Search API.
        Args:
            row_fields (Dict[str, Any]): The fields to format.

        Returns:
            Dict[str, Any]: The formatted metadata.
        """
    metadata = {}
    for key, value in row_fields.items():
        if key.startswith(self._metadata_key):
            new_key = key.split(self._metadata_key + '.')[-1]
            metadata[new_key] = value
        else:
            metadata[key] = value
    return metadata