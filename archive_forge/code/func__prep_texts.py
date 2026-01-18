from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
def _prep_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]], ids: Optional[List[str]]) -> List[dict]:
    """Embed and create the documents"""
    _ids = ids or (str(uuid.uuid4()) for _ in texts)
    _metadatas: Iterable[dict] = metadatas or ({} for _ in texts)
    embedded_texts = self._embedding.embed_documents(list(texts))
    return [{'id': _id, 'vec': vec, f'{self._text_key}': text, 'metadata': metadata} for _id, vec, text, metadata in zip(_ids, embedded_texts, texts, _metadatas)]