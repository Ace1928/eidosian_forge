from __future__ import annotations
import re
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra, Field
from langchain.callbacks.manager import Callbacks
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.llm import LLMChain
from langchain.evaluation.criteria.prompt import PROMPT, PROMPT_WITH_REFERENCES
from langchain.evaluation.schema import LLMEvalChain, StringEvaluator
from langchain.schema import RUN_KEY
@classmethod
def _resolve_prompt(cls, prompt: Optional[BasePromptTemplate]=None) -> BasePromptTemplate:
    expected_input_vars = {'input', 'output', 'criteria', 'reference'}
    prompt_ = prompt or PROMPT_WITH_REFERENCES
    if expected_input_vars != set(prompt_.input_variables):
        raise ValueError(f'Input variables should be {expected_input_vars}, but got {prompt_.input_variables}')
    return prompt_