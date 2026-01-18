import datetime
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
def _get_rescored_docs(self, docs_and_scores: Dict[Any, Tuple[Document, Optional[float]]]) -> List[Document]:
    current_time = datetime.datetime.now()
    rescored_docs = [(doc, self._get_combined_score(doc, relevance, current_time)) for doc, relevance in docs_and_scores.values()]
    rescored_docs.sort(key=lambda x: x[1], reverse=True)
    result = []
    for doc, _ in rescored_docs[:self.k]:
        buffered_doc = self.memory_stream[doc.metadata['buffer_idx']]
        buffered_doc.metadata['last_accessed_at'] = current_time
        result.append(buffered_doc)
    return result