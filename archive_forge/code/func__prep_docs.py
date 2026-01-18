from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _prep_docs(self, texts: Iterable[str], metadatas: Optional[List[dict]], ids: Optional[List[str]]) -> List[TigrisDocument]:
    embeddings: List[List[float]] = self._embed_fn.embed_documents(list(texts))
    docs: List[TigrisDocument] = []
    for t, m, e, _id in itertools.zip_longest(texts, metadatas or [], embeddings or [], ids or []):
        doc: TigrisDocument = {'text': t, 'embeddings': e or [], 'metadata': m or {}}
        if _id:
            doc['id'] = _id
        docs.append(doc)
    return docs