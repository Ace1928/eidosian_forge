from __future__ import annotations
import logging
import math
import warnings
from abc import ABC, abstractmethod
from typing import (
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.config import run_in_executor
class VectorStoreRetriever(BaseRetriever):
    """Base Retriever class for VectorStore."""
    vectorstore: VectorStore
    'VectorStore to use for retrieval.'
    search_type: str = 'similarity'
    'Type of search to perform. Defaults to "similarity".'
    search_kwargs: dict = Field(default_factory=dict)
    'Keyword arguments to pass to the search function.'
    allowed_search_types: ClassVar[Collection[str]] = ('similarity', 'similarity_score_threshold', 'mmr')

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        search_type = values['search_type']
        if search_type not in cls.allowed_search_types:
            raise ValueError(f'search_type of {search_type} not allowed. Valid values are: {cls.allowed_search_types}')
        if search_type == 'similarity_score_threshold':
            score_threshold = values['search_kwargs'].get('score_threshold')
            if score_threshold is None or not isinstance(score_threshold, float):
                raise ValueError('`score_threshold` is not specified with a float value(0~1) in `search_kwargs`.')
        return values

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        if self.search_type == 'similarity':
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == 'similarity_score_threshold':
            docs_and_similarities = self.vectorstore.similarity_search_with_relevance_scores(query, **self.search_kwargs)
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == 'mmr':
            docs = self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)
        else:
            raise ValueError(f'search_type of {self.search_type} not allowed.')
        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
        if self.search_type == 'similarity':
            docs = await self.vectorstore.asimilarity_search(query, **self.search_kwargs)
        elif self.search_type == 'similarity_score_threshold':
            docs_and_similarities = await self.vectorstore.asimilarity_search_with_relevance_scores(query, **self.search_kwargs)
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == 'mmr':
            docs = await self.vectorstore.amax_marginal_relevance_search(query, **self.search_kwargs)
        else:
            raise ValueError(f'search_type of {self.search_type} not allowed.')
        return docs

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        return await self.vectorstore.aadd_documents(documents, **kwargs)