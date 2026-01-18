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
Initialize the LabeledPairwiseStringEvalChain from an LLM.

        Args:
            llm (BaseLanguageModel): The LLM to use.
            prompt (PromptTemplate, optional): The prompt to use.
            criteria (Union[CRITERIA_TYPE, str], optional): The criteria to use.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            LabeledPairwiseStringEvalChain: The initialized LabeledPairwiseStringEvalChain.

        Raises:
            ValueError: If the input variables are not as expected.

        