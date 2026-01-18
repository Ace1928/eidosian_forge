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
def _validate_gcs_bucket(cls, gcs_bucket_name: str) -> str:
    """Validates the gcs_bucket_name as a bucket name.

        Args:
              gcs_bucket_name: The received bucket uri.

        Returns:
              A valid gcs_bucket_name or throws ValueError if full path is
              provided.
        """
    gcs_bucket_name = gcs_bucket_name.replace('gs://', '')
    if '/' in gcs_bucket_name:
        raise ValueError(f'The argument gcs_bucket_name should only be the bucket name. Received {gcs_bucket_name}')
    return gcs_bucket_name