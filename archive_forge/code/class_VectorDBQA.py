from __future__ import annotations
import inspect
import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
class VectorDBQA(BaseRetrievalQA):
    """Chain for question-answering against a vector database."""
    vectorstore: VectorStore = Field(exclude=True, alias='vectorstore')
    'Vector Database to connect to.'
    k: int = 4
    'Number of documents to query for.'
    search_type: str = 'similarity'
    'Search type to use over vectorstore. `similarity` or `mmr`.'
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    'Extra search args.'

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        warnings.warn('`VectorDBQA` is deprecated - please use `from langchain.chains import RetrievalQA`')
        return values

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        if 'search_type' in values:
            search_type = values['search_type']
            if search_type not in ('similarity', 'mmr'):
                raise ValueError(f'search_type of {search_type} not allowed.')
        return values

    def _get_docs(self, question: str, *, run_manager: CallbackManagerForChainRun) -> List[Document]:
        """Get docs."""
        if self.search_type == 'similarity':
            docs = self.vectorstore.similarity_search(question, k=self.k, **self.search_kwargs)
        elif self.search_type == 'mmr':
            docs = self.vectorstore.max_marginal_relevance_search(question, k=self.k, **self.search_kwargs)
        else:
            raise ValueError(f'search_type of {self.search_type} not allowed.')
        return docs

    async def _aget_docs(self, question: str, *, run_manager: AsyncCallbackManagerForChainRun) -> List[Document]:
        """Get docs."""
        raise NotImplementedError('VectorDBQA does not support async')

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return 'vector_db_qa'