from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _mmr_search(self, query_emb: np.ndarray) -> List[Document]:
    """
        Perform a maximal marginal relevance (mmr) search.

        Args:
            query_emb: Query represented as an embedding

        Returns:
            A list of diverse documents related to the query
        """
    docs = self._search(query_emb=query_emb, top_k=20)
    mmr_selected = maximal_marginal_relevance(query_emb, [doc[self.search_field] if isinstance(doc, dict) else getattr(doc, self.search_field) for doc in docs], k=self.top_k)
    results = [self._docarray_to_langchain_doc(docs[idx]) for idx in mmr_selected]
    return results