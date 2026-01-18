from __future__ import annotations
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
Create an AwaDB vectorstore from a list of documents.

        If a log_and_data_dir specified, the table will be persisted there.

        Args:
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            table_name (str): Name of the table to create.
            log_and_data_dir (Optional[str]): Directory to persist the table.
            client (Optional[awadb.Client]): AwaDB client.
            Any: Any possible parameters in the future

        Returns:
            AwaDB: AwaDB vectorstore.
        