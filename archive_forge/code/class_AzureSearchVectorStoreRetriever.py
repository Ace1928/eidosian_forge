from __future__ import annotations
import base64
import json
import logging
import uuid
from typing import (
import numpy as np
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
class AzureSearchVectorStoreRetriever(BaseRetriever):
    """Retriever that uses `Azure Cognitive Search`."""
    vectorstore: AzureSearch
    'Azure Search instance used to find similar documents.'
    search_type: str = 'hybrid'
    'Type of search to perform. Options are "similarity", "hybrid",\n    "semantic_hybrid".'
    k: int = 4
    'Number of documents to return.'

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        if 'search_type' in values:
            search_type = values['search_type']
            if search_type not in (allowed_search_types := ('similarity', 'hybrid', 'semantic_hybrid')):
                raise ValueError(f'search_type of {search_type} not allowed. Valid values are: {allowed_search_types}')
        return values

    def _get_relevant_documents(self, query: str, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
        if self.search_type == 'similarity':
            docs = self.vectorstore.vector_search(query, k=self.k, **kwargs)
        elif self.search_type == 'hybrid':
            docs = self.vectorstore.hybrid_search(query, k=self.k, **kwargs)
        elif self.search_type == 'semantic_hybrid':
            docs = self.vectorstore.semantic_hybrid_search(query, k=self.k, **kwargs)
        else:
            raise ValueError(f'search_type of {self.search_type} not allowed.')
        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
        raise NotImplementedError('AzureSearchVectorStoreRetriever does not support async')