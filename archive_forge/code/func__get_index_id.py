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
def _get_index_id(self) -> str:
    """Gets the correct index id for the endpoint.

        Returns:
            The index id if found (which should be found) or throws
            ValueError otherwise.
        """
    for index in self.endpoint.deployed_indexes:
        if index.index == self.index.resource_name:
            return index.id
    raise ValueError(f'No index with id {self.index.resource_name} deployed on endpoint {self.endpoint.display_name}.')