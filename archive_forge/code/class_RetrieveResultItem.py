import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
class RetrieveResultItem(ResultItem):
    """Retrieve API result item."""
    DocumentTitle: Optional[str]
    'The document title.'
    Content: Optional[str]
    'The content of the item.'

    def get_title(self) -> str:
        return self.DocumentTitle or ''

    def get_excerpt(self) -> str:
        return self.Content or ''