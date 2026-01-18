from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from hashlib import md5
from typing import Any, Iterable, List, Optional, Tuple, Type
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
class VectaraRetriever(VectorStoreRetriever):
    """Retriever for `Vectara`."""
    vectorstore: Vectara
    'Vectara vectorstore.'
    search_kwargs: dict = Field(default_factory=lambda: {'lambda_val': 0.0, 'k': 5, 'filter': '', 'n_sentence_context': '2', 'summary_config': SummaryConfig()})
    'Search params.\n        k: Number of Documents to return. Defaults to 5.\n        lambda_val: lexical match parameter for hybrid search.\n        filter: Dictionary of argument(s) to filter on metadata. For example a\n            filter can be "doc.rating > 3.0 and part.lang = \'deu\'"} see\n            https://docs.vectara.com/docs/search-apis/sql/filter-overview\n            for more details.\n        n_sentence_context: number of sentences before/after the matching segment to add\n    '

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]]=None, doc_metadata: Optional[dict]=None) -> None:
        """Add text to the Vectara vectorstore.

        Args:
            texts (List[str]): The text
            metadatas (List[dict]): Metadata dicts, must line up with existing store
        """
        self.vectorstore.add_texts(texts, metadatas, doc_metadata or {})