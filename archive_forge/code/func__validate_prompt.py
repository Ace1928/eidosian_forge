from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import create_model
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains.base import Chain
def _validate_prompt(prompt: BasePromptTemplate) -> None:
    if DOCUMENTS_KEY not in prompt.input_variables:
        raise ValueError(f'Prompt must accept {DOCUMENTS_KEY} as an input variable. Received prompt with input variables: {prompt.input_variables}')