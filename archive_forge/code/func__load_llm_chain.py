import json
from pathlib import Path
from typing import Any, Union
import yaml
from langchain_community.llms.loading import load_llm, load_llm_from_config
from langchain_core.prompts.loading import (
from langchain.chains import ReduceDocumentsChain
from langchain.chains.api.base import APIChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.chains.llm import LLMChain
from langchain.chains.llm_checker.base import LLMCheckerChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains.llm_requests import LLMRequestsChain
from langchain.chains.qa_with_sources.base import QAWithSourcesChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.vector_db import VectorDBQAWithSourcesChain
from langchain.chains.retrieval_qa.base import RetrievalQA, VectorDBQA
def _load_llm_chain(config: dict, **kwargs: Any) -> LLMChain:
    """Load LLM chain from config dict."""
    if 'llm' in config:
        llm_config = config.pop('llm')
        llm = load_llm_from_config(llm_config, **kwargs)
    elif 'llm_path' in config:
        llm = load_llm(config.pop('llm_path'), **kwargs)
    else:
        raise ValueError('One of `llm` or `llm_path` must be present.')
    if 'prompt' in config:
        prompt_config = config.pop('prompt')
        prompt = load_prompt_from_config(prompt_config)
    elif 'prompt_path' in config:
        prompt = load_prompt(config.pop('prompt_path'))
    else:
        raise ValueError('One of `prompt` or `prompt_path` must be present.')
    _load_output_parser(config)
    return LLMChain(llm=llm, prompt=prompt, **config)