from __future__ import annotations
import logging
import warnings
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _generate_documents_to_add(self, texts: Iterable[str], metadatas: Optional[List[Dict[Any, Any]]]=None, document_ids: Optional[List[str]]=None) -> List[ZepDocument]:
    from zep_python.document import Document as ZepDocument
    embeddings = None
    if self._collection and self._collection.is_auto_embedded:
        if self._embedding is not None:
            warnings.warn('The collection is set to auto-embed and an embedding \n                function is present. Ignoring the embedding function.', stacklevel=2)
    elif self._embedding is not None:
        embeddings = self._embedding.embed_documents(list(texts))
        if self._collection and self._collection.embedding_dimensions != len(embeddings[0]):
            raise ValueError(f'The embedding dimensions of the collection and the embedding function do not match. Collection dimensions: {self._collection.embedding_dimensions}, Embedding dimensions: {len(embeddings[0])}')
    else:
        pass
    documents: List[ZepDocument] = []
    for i, d in enumerate(texts):
        documents.append(ZepDocument(content=d, metadata=metadatas[i] if metadatas else None, document_id=document_ids[i] if document_ids else None, embedding=embeddings[i] if embeddings else None))
    return documents