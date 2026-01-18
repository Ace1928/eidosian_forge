from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import (
from langchain.chains.llm import LLMChain
def _log_prepared_sparql_query(self, _run_manager: CallbackManagerForChainRun, generated_query: str) -> None:
    _run_manager.on_text('Generated SPARQL:', end='\n', verbose=self.verbose)
    _run_manager.on_text(generated_query, color='green', end='\n', verbose=self.verbose)