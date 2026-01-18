from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
from langchain_core.documents import Document
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.pydantic_v1 import BaseModel, Extra
from langchain_core.vectorstores import VectorStore
class _VectorStoreExampleSelector(BaseExampleSelector, BaseModel, ABC):
    """Example selector that selects examples based on SemanticSimilarity."""
    vectorstore: VectorStore
    'VectorStore than contains information about examples.'
    k: int = 4
    'Number of examples to select.'
    example_keys: Optional[List[str]] = None
    'Optional keys to filter examples to.'
    input_keys: Optional[List[str]] = None
    'Optional keys to filter input to. If provided, the search is based on\n    the input variables instead of all variables.'
    vectorstore_kwargs: Optional[Dict[str, Any]] = None
    'Extra arguments passed to similarity_search function of the vectorstore.'

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @staticmethod
    def _example_to_text(example: Dict[str, str], input_keys: Optional[List[str]]) -> str:
        if input_keys:
            return ' '.join(sorted_values({key: example[key] for key in input_keys}))
        else:
            return ' '.join(sorted_values(example))

    def _documents_to_examples(self, documents: List[Document]) -> List[dict]:
        examples = [dict(e.metadata) for e in documents]
        if self.example_keys:
            examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
        return examples

    def add_example(self, example: Dict[str, str]) -> str:
        """Add new example to vectorstore."""
        ids = self.vectorstore.add_texts([self._example_to_text(example, self.input_keys)], metadatas=[example])
        return ids[0]

    async def aadd_example(self, example: Dict[str, str]) -> str:
        """Add new example to vectorstore."""
        ids = await self.vectorstore.aadd_texts([self._example_to_text(example, self.input_keys)], metadatas=[example])
        return ids[0]