from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Union
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field
from langchain.callbacks.manager import Callbacks
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.llm import LLMChain
from langchain.evaluation.criteria.eval_chain import (
from langchain.evaluation.schema import LLMEvalChain, StringEvaluator
from langchain.evaluation.scoring.prompt import (
from langchain.schema import RUN_KEY
class LabeledScoreStringEvalChain(ScoreStringEvalChain):
    """A chain for scoring the output of a model on a scale of 1-10.

    Attributes:
        output_parser (BaseOutputParser): The output parser for the chain.

    """

    @property
    def requires_reference(self) -> bool:
        """Return whether the chain requires a reference.

        Returns:
            bool: True if the chain requires a reference, False otherwise.

        """
        return True

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, *, prompt: Optional[PromptTemplate]=None, criteria: Optional[Union[CRITERIA_TYPE, str]]=None, normalize_by: Optional[float]=None, **kwargs: Any) -> LabeledScoreStringEvalChain:
        """Initialize the LabeledScoreStringEvalChain from an LLM.

        Args:
            llm (BaseLanguageModel): The LLM to use.
            prompt (PromptTemplate, optional): The prompt to use.
            criteria (Union[CRITERIA_TYPE, str], optional): The criteria to use.
            normalize_by (float, optional): The value to normalize the score by.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            LabeledScoreStringEvalChain: The initialized LabeledScoreStringEvalChain.

        Raises:
            ValueError: If the input variables are not as expected.

        """
        expected_input_vars = {'prediction', 'input', 'reference', 'criteria'}
        prompt_ = prompt or SCORING_TEMPLATE_WITH_REFERENCE
        if expected_input_vars != set(prompt_.input_variables):
            raise ValueError(f'Input variables should be {expected_input_vars}, but got {prompt_.input_variables}')
        criteria_ = resolve_criteria(criteria)
        criteria_str = '\n'.join((f'{k}: {v}' for k, v in criteria_.items())).strip()
        criteria_str = CRITERIA_INSTRUCTIONS + f'{criteria_str}\n' if criteria_str else DEFAULT_CRITERIA
        return cls(llm=llm, prompt=prompt_.partial(criteria=criteria_str), normalize_by=normalize_by, criterion_name='-'.join(criteria_), **kwargs)