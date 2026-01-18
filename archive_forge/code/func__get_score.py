from __future__ import annotations
import re
import string
from typing import Any, List, Optional, Sequence, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Extra
from langchain.callbacks.manager import Callbacks
from langchain.chains.llm import LLMChain
from langchain.evaluation.qa.eval_prompt import CONTEXT_PROMPT, COT_PROMPT, PROMPT
from langchain.evaluation.schema import LLMEvalChain, StringEvaluator
from langchain.schema import RUN_KEY
def _get_score(text: str) -> Optional[Tuple[str, int]]:
    match = re.search('grade:\\s*(correct|incorrect)', text.strip(), re.IGNORECASE)
    if match:
        if match.group(1).upper() == 'CORRECT':
            return ('CORRECT', 1)
        elif match.group(1).upper() == 'INCORRECT':
            return ('INCORRECT', 0)
    try:
        first_word = text.strip().split()[0].translate(str.maketrans('', '', string.punctuation))
        if first_word.upper() == 'CORRECT':
            return ('CORRECT', 1)
        elif first_word.upper() == 'INCORRECT':
            return ('INCORRECT', 0)
        last_word = text.strip().split()[-1].translate(str.maketrans('', '', string.punctuation))
        if last_word.upper() == 'CORRECT':
            return ('CORRECT', 1)
        elif last_word.upper() == 'INCORRECT':
            return ('INCORRECT', 0)
    except IndexError:
        pass
    return None