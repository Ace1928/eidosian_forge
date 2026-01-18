from typing import Any, List, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import (
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import (
class PydanticSchema(BaseModel):
    info: List[pydantic_schema]