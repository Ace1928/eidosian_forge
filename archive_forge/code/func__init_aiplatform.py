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
def _init_aiplatform(cls, project_id: str, region: str, gcs_bucket_name: str, credentials: 'Credentials') -> None:
    """Configures the aiplatform library.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
            the same location as the GCS bucket and must be regional.
            gcs_bucket_name: GCS staging location.
            credentials: The GCS Credentials object.
        """
    from google.cloud import aiplatform
    logger.debug(f'Initializing AI Platform for project {project_id} on {region} and for {gcs_bucket_name}.')
    aiplatform.init(project=project_id, location=region, staging_bucket=gcs_bucket_name, credentials=credentials)