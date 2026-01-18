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
def _load_refine_chain(llm: BaseLanguageModel, question_prompt: BasePromptTemplate=refine_prompts.PROMPT, refine_prompt: BasePromptTemplate=refine_prompts.REFINE_PROMPT, document_variable_name: str='text', initial_response_name: str='existing_answer', refine_llm: Optional[BaseLanguageModel]=None, verbose: Optional[bool]=None, **kwargs: Any) -> RefineDocumentsChain:
    initial_chain = LLMChain(llm=llm, prompt=question_prompt, verbose=verbose)
    _refine_llm = refine_llm or llm
    refine_chain = LLMChain(llm=_refine_llm, prompt=refine_prompt, verbose=verbose)
    return RefineDocumentsChain(initial_llm_chain=initial_chain, refine_llm_chain=refine_chain, document_variable_name=document_variable_name, initial_response_name=initial_response_name, verbose=verbose, **kwargs)