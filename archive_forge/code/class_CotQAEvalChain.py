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
class CotQAEvalChain(ContextQAEvalChain):
    """LLM Chain for evaluating QA using chain of thought reasoning."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def evaluation_name(self) -> str:
        return 'COT Contextual Accuracy'

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, prompt: Optional[PromptTemplate]=None, **kwargs: Any) -> CotQAEvalChain:
        """Load QA Eval Chain from LLM."""
        prompt = prompt or COT_PROMPT
        cls._validate_input_vars(prompt)
        return cls(llm=llm, prompt=prompt, **kwargs)