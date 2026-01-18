from __future__ import annotations
import inspect
import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
class BaseRetrievalQA(Chain):
    """Base class for question-answering chains."""
    combine_documents_chain: BaseCombineDocumentsChain
    'Chain to use to combine the documents.'
    input_key: str = 'query'
    output_key: str = 'result'
    return_source_documents: bool = False
    'Return the source documents or not.'

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ['source_documents']
        return _output_keys

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, prompt: Optional[PromptTemplate]=None, callbacks: Callbacks=None, llm_chain_kwargs: Optional[dict]=None, **kwargs: Any) -> BaseRetrievalQA:
        """Initialize from LLM."""
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        llm_chain = LLMChain(llm=llm, prompt=_prompt, callbacks=callbacks, **llm_chain_kwargs or {})
        document_prompt = PromptTemplate(input_variables=['page_content'], template='Context:\n{page_content}')
        combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='context', document_prompt=document_prompt, callbacks=callbacks)
        return cls(combine_documents_chain=combine_documents_chain, callbacks=callbacks, **kwargs)

    @classmethod
    def from_chain_type(cls, llm: BaseLanguageModel, chain_type: str='stuff', chain_type_kwargs: Optional[dict]=None, **kwargs: Any) -> BaseRetrievalQA:
        """Load chain from chain type."""
        _chain_type_kwargs = chain_type_kwargs or {}
        combine_documents_chain = load_qa_chain(llm, chain_type=chain_type, **_chain_type_kwargs)
        return cls(combine_documents_chain=combine_documents_chain, **kwargs)

    @abstractmethod
    def _get_docs(self, question: str, *, run_manager: CallbackManagerForChainRun) -> List[Document]:
        """Get documents to do question answering over."""

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        accepts_run_manager = 'run_manager' in inspect.signature(self._get_docs).parameters
        if accepts_run_manager:
            docs = self._get_docs(question, run_manager=_run_manager)
        else:
            docs = self._get_docs(question)
        answer = self.combine_documents_chain.run(input_documents=docs, question=question, callbacks=_run_manager.get_child())
        if self.return_source_documents:
            return {self.output_key: answer, 'source_documents': docs}
        else:
            return {self.output_key: answer}

    @abstractmethod
    async def _aget_docs(self, question: str, *, run_manager: AsyncCallbackManagerForChainRun) -> List[Document]:
        """Get documents to do question answering over."""

    async def _acall(self, inputs: Dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        accepts_run_manager = 'run_manager' in inspect.signature(self._aget_docs).parameters
        if accepts_run_manager:
            docs = await self._aget_docs(question, run_manager=_run_manager)
        else:
            docs = await self._aget_docs(question)
        answer = await self.combine_documents_chain.arun(input_documents=docs, question=question, callbacks=_run_manager.get_child())
        if self.return_source_documents:
            return {self.output_key: answer, 'source_documents': docs}
        else:
            return {self.output_key: answer}