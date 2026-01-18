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
def _prepare_sparql_query(self, _run_manager: CallbackManagerForChainRun, generated_sparql: str) -> str:
    from rdflib.plugins.sparql import prepareQuery
    prepareQuery(generated_sparql)
    self._log_prepared_sparql_query(_run_manager, generated_sparql)
    return generated_sparql