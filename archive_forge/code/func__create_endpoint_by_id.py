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
def _create_endpoint_by_id(cls, endpoint_id: str, project_id: str, region: str, credentials: 'Credentials') -> MatchingEngineIndexEndpoint:
    """Creates a MatchingEngineIndexEndpoint object by id.

        Args:
            endpoint_id: The created endpoint id.
            project_id: The project to retrieve index from.
            region: Location to retrieve index from.
            credentials: GCS credentials.

        Returns:
            A configured MatchingEngineIndexEndpoint.
        """
    from google.cloud import aiplatform
    logger.debug(f'Creating endpoint with id {endpoint_id}.')
    return aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id, project=project_id, location=region, credentials=credentials)