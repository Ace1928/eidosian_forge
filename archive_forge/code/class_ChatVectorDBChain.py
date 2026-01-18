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
class ChatVectorDBChain(BaseConversationalRetrievalChain):
    """Chain for chatting with a vector database."""
    vectorstore: VectorStore = Field(alias='vectorstore')
    top_k_docs_for_context: int = 4
    search_kwargs: dict = Field(default_factory=dict)

    @property
    def _chain_type(self) -> str:
        return 'chat-vector-db'

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        warnings.warn('`ChatVectorDBChain` is deprecated - please use `from langchain.chains import ConversationalRetrievalChain`')
        return values

    def _get_docs(self, question: str, inputs: Dict[str, Any], *, run_manager: CallbackManagerForChainRun) -> List[Document]:
        """Get docs."""
        vectordbkwargs = inputs.get('vectordbkwargs', {})
        full_kwargs = {**self.search_kwargs, **vectordbkwargs}
        return self.vectorstore.similarity_search(question, k=self.top_k_docs_for_context, **full_kwargs)

    async def _aget_docs(self, question: str, inputs: Dict[str, Any], *, run_manager: AsyncCallbackManagerForChainRun) -> List[Document]:
        """Get docs."""
        raise NotImplementedError('ChatVectorDBChain does not support async')

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, vectorstore: VectorStore, condense_question_prompt: BasePromptTemplate=CONDENSE_QUESTION_PROMPT, chain_type: str='stuff', combine_docs_chain_kwargs: Optional[Dict]=None, callbacks: Callbacks=None, **kwargs: Any) -> BaseConversationalRetrievalChain:
        """Load chain from LLM."""
        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {}
        doc_chain = load_qa_chain(llm, chain_type=chain_type, callbacks=callbacks, **combine_docs_chain_kwargs)
        condense_question_chain = LLMChain(llm=llm, prompt=condense_question_prompt, callbacks=callbacks)
        return cls(vectorstore=vectorstore, combine_docs_chain=doc_chain, question_generator=condense_question_chain, callbacks=callbacks, **kwargs)