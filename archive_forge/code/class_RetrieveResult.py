import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
class RetrieveResult(BaseModel, extra=Extra.allow):
    """`Amazon Kendra Retrieve API` search result.

    It is composed of:
        * relevant passages or text excerpts given an input query.
    """
    QueryId: str
    'The ID of the query.'
    ResultItems: List[RetrieveResultItem]
    'The result items.'