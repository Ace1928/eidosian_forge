from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
from langchain_core.documents import Document
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.pydantic_v1 import BaseModel, Extra
from langchain_core.vectorstores import VectorStore
class SemanticSimilarityExampleSelector(_VectorStoreExampleSelector):
    """Example selector that selects examples based on SemanticSimilarity."""

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on semantic similarity."""
        vectorstore_kwargs = self.vectorstore_kwargs or {}
        example_docs = self.vectorstore.similarity_search(self._example_to_text(input_variables, self.input_keys), k=self.k, **vectorstore_kwargs)
        return self._documents_to_examples(example_docs)

    async def aselect_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on semantic similarity."""
        vectorstore_kwargs = self.vectorstore_kwargs or {}
        example_docs = await self.vectorstore.asimilarity_search(self._example_to_text(input_variables, self.input_keys), k=self.k, **vectorstore_kwargs)
        return self._documents_to_examples(example_docs)

    @classmethod
    def from_examples(cls, examples: List[dict], embeddings: Embeddings, vectorstore_cls: Type[VectorStore], k: int=4, input_keys: Optional[List[str]]=None, *, example_keys: Optional[List[str]]=None, vectorstore_kwargs: Optional[dict]=None, **vectorstore_cls_kwargs: Any) -> SemanticSimilarityExampleSelector:
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the vectorstore.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs)
        return cls(vectorstore=vectorstore, k=k, input_keys=input_keys, example_keys=example_keys, vectorstore_kwargs=vectorstore_kwargs)

    @classmethod
    async def afrom_examples(cls, examples: List[dict], embeddings: Embeddings, vectorstore_cls: Type[VectorStore], k: int=4, input_keys: Optional[List[str]]=None, *, example_keys: Optional[List[str]]=None, vectorstore_kwargs: Optional[dict]=None, **vectorstore_cls_kwargs: Any) -> SemanticSimilarityExampleSelector:
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the vectorstore.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = await vectorstore_cls.afrom_texts(string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs)
        return cls(vectorstore=vectorstore, k=k, input_keys=input_keys, example_keys=example_keys, vectorstore_kwargs=vectorstore_kwargs)