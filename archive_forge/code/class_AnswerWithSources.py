from typing import Any, List, Optional, Type, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseLLMOutputParser
from langchain_core.output_parsers.openai_functions import (
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import get_llm_kwargs
class AnswerWithSources(BaseModel):
    """An answer to the question, with sources."""
    answer: str = Field(..., description='Answer to the question that was asked')
    sources: List[str] = Field(..., description='List of sources used to answer the question')