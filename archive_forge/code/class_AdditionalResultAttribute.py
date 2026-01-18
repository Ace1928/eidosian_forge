import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
class AdditionalResultAttribute(BaseModel, extra=Extra.allow):
    """Additional result attribute."""
    Key: str
    'The key of the attribute.'
    ValueType: Literal['TEXT_WITH_HIGHLIGHTS_VALUE']
    'The type of the value.'
    Value: AdditionalResultAttributeValue
    'The value of the attribute.'

    def get_value_text(self) -> str:
        return self.Value.TextWithHighlightsValue.Text