from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from langchain_community.graphs.arangodb_graph import ArangoGraph
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import (
from langchain.chains.llm import LLMChain

        Generate an AQL statement from user input, use it retrieve a response
        from an ArangoDB Database instance, and respond to the user input
        in natural language.

        Users can modify the following ArangoGraphQAChain Class Variables:

        :var top_k: The maximum number of AQL Query Results to return
        :type top_k: int

        :var aql_examples: A set of AQL Query Examples that are passed to
            the AQL Generation Prompt Template to promote few-shot-learning.
            Defaults to an empty string.
        :type aql_examples: str

        :var return_aql_query: Whether to return the AQL Query in the
            output dictionary. Defaults to False.
        :type return_aql_query: bool

        :var return_aql_result: Whether to return the AQL Query in the
            output dictionary. Defaults to False
        :type return_aql_result: bool

        :var max_aql_generation_attempts: The maximum amount of AQL
            Generation attempts to be made prior to raising the last
            AQL Query Execution Error. Defaults to 3.
        :type max_aql_generation_attempts: int
        