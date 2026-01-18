import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
class QueryResultItem(ResultItem):
    """Query API result item."""
    DocumentTitle: TextWithHighLights
    'The document title.'
    FeedbackToken: Optional[str]
    'Identifies a particular result from a particular query.'
    Format: Optional[str]
    '\n    If the Type is ANSWER, then format is either:\n        * TABLE: a table excerpt is returned in TableExcerpt;\n        * TEXT: a text excerpt is returned in DocumentExcerpt.\n    '
    Type: Optional[str]
    'Type of result: DOCUMENT or QUESTION_ANSWER or ANSWER'
    AdditionalAttributes: Optional[List[AdditionalResultAttribute]] = []
    'One or more additional attributes associated with the result.'
    DocumentExcerpt: Optional[TextWithHighLights]
    'Excerpt of the document text.'

    def get_title(self) -> str:
        return self.DocumentTitle.Text

    def get_attribute_value(self) -> str:
        if not self.AdditionalAttributes:
            return ''
        if not self.AdditionalAttributes[0]:
            return ''
        else:
            return self.AdditionalAttributes[0].get_value_text()

    def get_excerpt(self) -> str:
        if self.AdditionalAttributes and self.AdditionalAttributes[0].Key == 'AnswerText':
            excerpt = self.get_attribute_value()
        elif self.DocumentExcerpt:
            excerpt = self.DocumentExcerpt.Text
        else:
            excerpt = ''
        return excerpt

    def get_additional_metadata(self) -> dict:
        additional_metadata = {'type': self.Type}
        return additional_metadata