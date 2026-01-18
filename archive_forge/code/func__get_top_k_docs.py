import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
def _get_top_k_docs(self, result_items: Sequence[ResultItem]) -> List[Document]:
    top_docs = [item.to_doc(self.page_content_formatter) for item in result_items[:self.top_k]]
    return top_docs