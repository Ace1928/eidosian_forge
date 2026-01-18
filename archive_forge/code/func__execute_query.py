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
def _execute_query(self, query: str) -> List[rdflib.query.ResultRow]:
    try:
        return self.graph.query(query)
    except Exception:
        raise ValueError('Failed to execute the generated SPARQL query.')