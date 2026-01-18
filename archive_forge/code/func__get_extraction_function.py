from typing import Any, List, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import (
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import (
def _get_extraction_function(entity_schema: dict) -> dict:
    return {'name': 'information_extraction', 'description': 'Extracts the relevant information from the passage.', 'parameters': {'type': 'object', 'properties': {'info': {'type': 'array', 'items': _convert_schema(entity_schema)}}, 'required': ['info']}}