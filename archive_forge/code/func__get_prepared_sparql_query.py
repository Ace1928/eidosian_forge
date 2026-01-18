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
def _get_prepared_sparql_query(self, _run_manager: CallbackManagerForChainRun, callbacks: CallbackManager, generated_sparql: str, ontology_schema: str) -> str:
    try:
        return self._prepare_sparql_query(_run_manager, generated_sparql)
    except Exception as e:
        retries = 0
        error_message = str(e)
        self._log_invalid_sparql_query(_run_manager, generated_sparql, error_message)
        while retries < self.max_fix_retries:
            try:
                sparql_fix_chain_result = self.sparql_fix_chain.invoke({'error_message': error_message, 'generated_sparql': generated_sparql, 'schema': ontology_schema}, callbacks=callbacks)
                generated_sparql = sparql_fix_chain_result[self.sparql_fix_chain.output_key]
                return self._prepare_sparql_query(_run_manager, generated_sparql)
            except Exception as e:
                retries += 1
                parse_exception = str(e)
                self._log_invalid_sparql_query(_run_manager, generated_sparql, parse_exception)
    raise ValueError('The generated SPARQL query is invalid.')