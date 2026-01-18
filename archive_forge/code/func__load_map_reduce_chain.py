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
def _load_map_reduce_chain(llm: BaseLanguageModel, map_prompt: BasePromptTemplate=map_reduce_prompt.PROMPT, combine_prompt: BasePromptTemplate=map_reduce_prompt.PROMPT, combine_document_variable_name: str='text', map_reduce_document_variable_name: str='text', collapse_prompt: Optional[BasePromptTemplate]=None, reduce_llm: Optional[BaseLanguageModel]=None, collapse_llm: Optional[BaseLanguageModel]=None, verbose: Optional[bool]=None, token_max: int=3000, callbacks: Callbacks=None, *, collapse_max_retries: Optional[int]=None, **kwargs: Any) -> MapReduceDocumentsChain:
    map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=verbose, callbacks=callbacks)
    _reduce_llm = reduce_llm or llm
    reduce_chain = LLMChain(llm=_reduce_llm, prompt=combine_prompt, verbose=verbose, callbacks=callbacks)
    combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name=combine_document_variable_name, verbose=verbose, callbacks=callbacks)
    if collapse_prompt is None:
        collapse_chain = None
        if collapse_llm is not None:
            raise ValueError('collapse_llm provided, but collapse_prompt was not: please provide one or stop providing collapse_llm.')
    else:
        _collapse_llm = collapse_llm or llm
        collapse_chain = StuffDocumentsChain(llm_chain=LLMChain(llm=_collapse_llm, prompt=collapse_prompt, verbose=verbose, callbacks=callbacks), document_variable_name=combine_document_variable_name)
    reduce_documents_chain = ReduceDocumentsChain(combine_documents_chain=combine_documents_chain, collapse_documents_chain=collapse_chain, token_max=token_max, verbose=verbose, callbacks=callbacks, collapse_max_retries=collapse_max_retries)
    return MapReduceDocumentsChain(llm_chain=map_chain, reduce_documents_chain=reduce_documents_chain, document_variable_name=map_reduce_document_variable_name, verbose=verbose, callbacks=callbacks, **kwargs)