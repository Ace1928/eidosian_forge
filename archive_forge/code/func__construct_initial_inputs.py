from __future__ import annotations
from typing import Any, Dict, List, Tuple
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain.chains.combine_documents.base import (
from langchain.chains.llm import LLMChain
def _construct_initial_inputs(self, docs: List[Document], **kwargs: Any) -> Dict[str, Any]:
    base_info = {'page_content': docs[0].page_content}
    base_info.update(docs[0].metadata)
    document_info = {k: base_info[k] for k in self.document_prompt.input_variables}
    base_inputs: dict = {self.document_variable_name: self.document_prompt.format(**document_info)}
    inputs = {**base_inputs, **kwargs}
    return inputs