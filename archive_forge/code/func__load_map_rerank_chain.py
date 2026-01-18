from typing import Any, Mapping, Optional, Protocol
from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain.chains import ReduceDocumentsChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import (
from langchain.chains.question_answering.map_rerank_prompt import (
def _load_map_rerank_chain(llm: BaseLanguageModel, prompt: BasePromptTemplate=MAP_RERANK_PROMPT, verbose: bool=False, document_variable_name: str='context', rank_key: str='score', answer_key: str='answer', callback_manager: Optional[BaseCallbackManager]=None, callbacks: Callbacks=None, **kwargs: Any) -> MapRerankDocumentsChain:
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose, callback_manager=callback_manager, callbacks=callbacks)
    return MapRerankDocumentsChain(llm_chain=llm_chain, rank_key=rank_key, answer_key=answer_key, document_variable_name=document_variable_name, verbose=verbose, callback_manager=callback_manager, **kwargs)