from __future__ import annotations
import uuid
import warnings
from typing import (
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore

        Create a new ElasticKnnSearch instance and add a list of texts to the
            Elasticsearch index.

        Args:
            texts (List[str]): The texts to add to the index.
            embedding (Embeddings): The embedding model to use for transforming the
                texts into vectors.
            metadatas (List[Dict[Any, Any]], optional): A list of metadata dictionaries
                to associate with the texts.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A new ElasticKnnSearch instance.
        