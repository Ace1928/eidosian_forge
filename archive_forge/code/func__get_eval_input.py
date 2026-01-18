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
def _get_eval_input(self, prediction: str, reference: Optional[str], input: Optional[str]) -> dict:
    """Get the evaluation input."""
    input_ = {'input': input, 'output': prediction}
    if self.requires_reference:
        input_['reference'] = reference
    return input_