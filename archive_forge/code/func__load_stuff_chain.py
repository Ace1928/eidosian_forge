from typing import Any, Mapping, Optional, Protocol
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import map_reduce_prompt, refine_prompts, stuff_prompt
def _load_stuff_chain(llm: BaseLanguageModel, prompt: BasePromptTemplate=stuff_prompt.PROMPT, document_variable_name: str='text', verbose: Optional[bool]=None, **kwargs: Any) -> StuffDocumentsChain:
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    return StuffDocumentsChain(llm_chain=llm_chain, document_variable_name=document_variable_name, verbose=verbose, **kwargs)