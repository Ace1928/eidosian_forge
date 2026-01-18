from __future__ import annotations
import math
import re
import warnings
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.prompt import PROMPT
def _evaluate_expression(self, expression: str) -> str:
    import numexpr
    try:
        local_dict = {'pi': math.pi, 'e': math.e}
        output = str(numexpr.evaluate(expression.strip(), global_dict={}, local_dict=local_dict))
    except Exception as e:
        raise ValueError(f'LLMMathChain._evaluate("{expression}") raised error: {e}. Please try again with a valid numerical expression')
    return re.sub('^\\[|\\]$', '', output)