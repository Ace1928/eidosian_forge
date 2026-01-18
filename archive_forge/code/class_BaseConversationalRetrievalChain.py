from __future__ import annotations
import inspect
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
class BaseConversationalRetrievalChain(Chain):
    """Chain for chatting with an index."""
    combine_docs_chain: BaseCombineDocumentsChain
    'The chain used to combine any retrieved documents.'
    question_generator: LLMChain
    'The chain used to generate a new question for the sake of retrieval.\n    This chain will take in the current question (with variable `question`)\n    and any chat history (with variable `chat_history`) and will produce\n    a new standalone question to be used later on.'
    output_key: str = 'answer'
    'The output key to return the final answer of this chain in.'
    rephrase_question: bool = True
    'Whether or not to pass the new generated question to the combine_docs_chain.\n    If True, will pass the new generated question along.\n    If False, will only use the new generated question for retrieval and pass the\n    original question along to the combine_docs_chain.'
    return_source_documents: bool = False
    'Return the retrieved source documents as part of the final result.'
    return_generated_question: bool = False
    'Return the generated question as part of the final result.'
    get_chat_history: Optional[Callable[[List[CHAT_TURN_TYPE]], str]] = None
    'An optional function to get a string of the chat history.\n    If None is provided, will use a default.'
    response_if_no_docs_found: Optional[str]
    'If specified, the chain will return a fixed response if no docs \n    are found for the question. '

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return ['question', 'chat_history']

    def get_input_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        return InputType

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ['source_documents']
        if self.return_generated_question:
            _output_keys = _output_keys + ['generated_question']
        return _output_keys

    @abstractmethod
    def _get_docs(self, question: str, inputs: Dict[str, Any], *, run_manager: CallbackManagerForChainRun) -> List[Document]:
        """Get docs."""

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun]=None) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs['question']
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs['chat_history'])
        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = self.question_generator.run(question=question, chat_history=chat_history_str, callbacks=callbacks)
        else:
            new_question = question
        accepts_run_manager = 'run_manager' in inspect.signature(self._get_docs).parameters
        if accepts_run_manager:
            docs = self._get_docs(new_question, inputs, run_manager=_run_manager)
        else:
            docs = self._get_docs(new_question, inputs)
        output: Dict[str, Any] = {}
        if self.response_if_no_docs_found is not None and len(docs) == 0:
            output[self.output_key] = self.response_if_no_docs_found
        else:
            new_inputs = inputs.copy()
            if self.rephrase_question:
                new_inputs['question'] = new_question
            new_inputs['chat_history'] = chat_history_str
            answer = self.combine_docs_chain.run(input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs)
            output[self.output_key] = answer
        if self.return_source_documents:
            output['source_documents'] = docs
        if self.return_generated_question:
            output['generated_question'] = new_question
        return output

    @abstractmethod
    async def _aget_docs(self, question: str, inputs: Dict[str, Any], *, run_manager: AsyncCallbackManagerForChainRun) -> List[Document]:
        """Get docs."""

    async def _acall(self, inputs: Dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun]=None) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs['question']
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs['chat_history'])
        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = await self.question_generator.arun(question=question, chat_history=chat_history_str, callbacks=callbacks)
        else:
            new_question = question
        accepts_run_manager = 'run_manager' in inspect.signature(self._aget_docs).parameters
        if accepts_run_manager:
            docs = await self._aget_docs(new_question, inputs, run_manager=_run_manager)
        else:
            docs = await self._aget_docs(new_question, inputs)
        output: Dict[str, Any] = {}
        if self.response_if_no_docs_found is not None and len(docs) == 0:
            output[self.output_key] = self.response_if_no_docs_found
        else:
            new_inputs = inputs.copy()
            if self.rephrase_question:
                new_inputs['question'] = new_question
            new_inputs['chat_history'] = chat_history_str
            answer = await self.combine_docs_chain.arun(input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs)
            output[self.output_key] = answer
        if self.return_source_documents:
            output['source_documents'] = docs
        if self.return_generated_question:
            output['generated_question'] = new_question
        return output

    def save(self, file_path: Union[Path, str]) -> None:
        if self.get_chat_history:
            raise ValueError('Chain not saveable when `get_chat_history` is not None.')
        super().save(file_path)