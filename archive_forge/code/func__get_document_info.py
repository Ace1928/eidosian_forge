from __future__ import annotations
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
import yaml
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompt_values import (
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.runnables.utils import create_model
def _get_document_info(doc: Document, prompt: BasePromptTemplate[str]) -> Dict:
    base_info = {'page_content': doc.page_content, **doc.metadata}
    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        required_metadata = [iv for iv in prompt.input_variables if iv != 'page_content']
        raise ValueError(f'Document prompt requires documents to have metadata variables: {required_metadata}. Received document with missing metadata: {list(missing_metadata)}.')
    return {k: base_info[k] for k in prompt.input_variables}