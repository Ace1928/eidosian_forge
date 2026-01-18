from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseLLMOutputParser
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.chains.base import Chain
from langchain.chains.elasticsearch_database.prompts import ANSWER_PROMPT, DSL_PROMPT
from langchain.chains.llm import LLMChain
def _get_indices_infos(self, indices: List[str]) -> str:
    mappings = self.database.indices.get_mapping(index=','.join(indices))
    if self.sample_documents_in_index_info > 0:
        for k, v in mappings.items():
            hits = self.database.search(index=k, query={'match_all': {}}, size=self.sample_documents_in_index_info)['hits']['hits']
            hits = [str(hit['_source']) for hit in hits]
            mappings[k]['mappings'] = str(v) + '\n\n/*\n' + '\n'.join(hits) + '\n*/'
    return '\n\n'.join(['Mapping for index {}:\n{}'.format(index, mappings[index]['mappings']) for index in mappings])