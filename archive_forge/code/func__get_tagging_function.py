from typing import Any, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import (
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import _convert_schema, get_llm_kwargs
def _get_tagging_function(schema: dict) -> dict:
    return {'name': 'information_extraction', 'description': 'Extracts the relevant information from the passage.', 'parameters': _convert_schema(schema)}