from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_checker.prompt import (
from langchain.chains.sequential import SequentialChain
def _load_question_to_checked_assertions_chain(llm: BaseLanguageModel, create_draft_answer_prompt: PromptTemplate, list_assertions_prompt: PromptTemplate, check_assertions_prompt: PromptTemplate, revised_answer_prompt: PromptTemplate) -> SequentialChain:
    create_draft_answer_chain = LLMChain(llm=llm, prompt=create_draft_answer_prompt, output_key='statement')
    list_assertions_chain = LLMChain(llm=llm, prompt=list_assertions_prompt, output_key='assertions')
    check_assertions_chain = LLMChain(llm=llm, prompt=check_assertions_prompt, output_key='checked_assertions')
    revised_answer_chain = LLMChain(llm=llm, prompt=revised_answer_prompt, output_key='revised_statement')
    chains = [create_draft_answer_chain, list_assertions_chain, check_assertions_chain, revised_answer_chain]
    question_to_checked_assertions_chain = SequentialChain(chains=chains, input_variables=['question'], output_variables=['revised_statement'], verbose=True)
    return question_to_checked_assertions_chain