import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
class Highlight(BaseModel, extra=Extra.allow):
    """Information that highlights the keywords in the excerpt."""
    BeginOffset: int
    'The zero-based location in the excerpt where the highlight starts.'
    EndOffset: int
    'The zero-based location in the excerpt where the highlight ends.'
    TopAnswer: Optional[bool]
    'Indicates whether the result is the best one.'
    Type: Optional[str]
    'The highlight type: STANDARD or THESAURUS_SYNONYM.'