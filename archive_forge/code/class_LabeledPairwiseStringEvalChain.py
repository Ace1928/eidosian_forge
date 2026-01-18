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
from langchain.evaluation.comparison.prompt import (
from langchain.evaluation.criteria.eval_chain import (
from langchain.evaluation.schema import LLMEvalChain, PairwiseStringEvaluator
from langchain.schema import RUN_KEY
class LabeledPairwiseStringEvalChain(PairwiseStringEvalChain):
    """A chain for comparing two outputs, such as the outputs
     of two models, prompts, or outputs of a single model on similar inputs,
     with labeled preferences.

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
    def from_llm(cls, llm: BaseLanguageModel, *, prompt: Optional[PromptTemplate]=None, criteria: Optional[Union[CRITERIA_TYPE, str]]=None, **kwargs: Any) -> PairwiseStringEvalChain:
        """Initialize the LabeledPairwiseStringEvalChain from an LLM.

        Args:
            llm (BaseLanguageModel): The LLM to use.
            prompt (PromptTemplate, optional): The prompt to use.
            criteria (Union[CRITERIA_TYPE, str], optional): The criteria to use.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            LabeledPairwiseStringEvalChain: The initialized LabeledPairwiseStringEvalChain.

        Raises:
            ValueError: If the input variables are not as expected.

        """
        expected_input_vars = {'prediction', 'prediction_b', 'input', 'reference', 'criteria'}
        prompt_ = prompt or COMPARISON_TEMPLATE_WITH_REFERENCE
        if expected_input_vars != set(prompt_.input_variables):
            raise ValueError(f'Input variables should be {expected_input_vars}, but got {prompt_.input_variables}')
        criteria_ = resolve_pairwise_criteria(criteria)
        criteria_str = '\n'.join((f'{k}: {v}' for k, v in criteria_.items()))
        criteria_str = CRITERIA_INSTRUCTIONS + criteria_str if criteria_str else ''
        return cls(llm=llm, prompt=prompt_.partial(criteria=criteria_str), **kwargs)