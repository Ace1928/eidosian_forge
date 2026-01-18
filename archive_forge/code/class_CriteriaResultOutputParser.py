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
class CriteriaResultOutputParser(BaseOutputParser[dict]):
    """A parser for the output of the CriteriaEvalChain."""

    @property
    def _type(self) -> str:
        return 'criteria_result'

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the output text.

        Args:
            text (str): The output text to parse.

        Returns:
            Dict: The parsed output.
        """
        verdict = None
        score = None
        match_last = re.search('\\s*(Y|N)\\s*$', text, re.IGNORECASE)
        match_first = re.search('^\\s*(Y|N)\\s*', text, re.IGNORECASE)
        match_end = re.search('\\b(Y|N)\\b\\s*$', text, re.IGNORECASE)
        if match_last:
            verdict = match_last.group(1).strip()
            text = text[:match_last.start()].strip()
        elif match_first:
            verdict = match_first.group(1).strip()
            text = text[match_first.end():].strip()
        elif match_end:
            verdict = match_end.group(1).strip()
            text = text[:match_end.start()].strip()
        else:
            splits = text.strip().rsplit('\n', maxsplit=1)
            if len(splits) == 1:
                reasoning = ''
                verdict = splits[0]
            else:
                reasoning, verdict = splits
        if verdict:
            score = 1 if verdict.upper() == 'Y' else 0 if verdict.upper() == 'N' else None
        return {'reasoning': text.strip(), 'value': verdict, 'score': score}