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
def _process_llm_result(self, llm_output: str, run_manager: CallbackManagerForChainRun) -> Dict[str, str]:
    run_manager.on_text(llm_output, color='green', verbose=self.verbose)
    llm_output = llm_output.strip()
    text_match = re.search('^```text(.*?)```', llm_output, re.DOTALL)
    if text_match:
        expression = text_match.group(1)
        output = self._evaluate_expression(expression)
        run_manager.on_text('\nAnswer: ', verbose=self.verbose)
        run_manager.on_text(output, color='yellow', verbose=self.verbose)
        answer = 'Answer: ' + output
    elif llm_output.startswith('Answer:'):
        answer = llm_output
    elif 'Answer:' in llm_output:
        answer = 'Answer: ' + llm_output.split('Answer:')[-1]
    else:
        raise ValueError(f'unknown format from LLM: {llm_output}')
    return {self.output_key: answer}