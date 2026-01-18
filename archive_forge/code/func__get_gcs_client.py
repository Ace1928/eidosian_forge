from __future__ import annotations
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Type
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.utilities.vertexai import get_client_info
@classmethod
def _get_gcs_client(cls, credentials: 'Credentials', project_id: str) -> 'storage.Client':
    """Lazily creates a GCS client.

        Returns:
            A configured GCS client.
        """
    from google.cloud import storage
    return storage.Client(credentials=credentials, project=project_id, client_info=get_client_info(module='vertex-ai-matching-engine'))