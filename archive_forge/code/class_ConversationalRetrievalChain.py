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
class ConversationalRetrievalChain(BaseConversationalRetrievalChain):
    """Chain for having a conversation based on retrieved documents.

    This chain takes in chat history (a list of messages) and new questions,
    and then returns an answer to that question.
    The algorithm for this chain consists of three parts:

    1. Use the chat history and the new question to create a "standalone question".
    This is done so that this question can be passed into the retrieval step to fetch
    relevant documents. If only the new question was passed in, then relevant context
    may be lacking. If the whole conversation was passed into retrieval, there may
    be unnecessary information there that would distract from retrieval.

    2. This new question is passed to the retriever and relevant documents are
    returned.

    3. The retrieved documents are passed to an LLM along with either the new question
    (default behavior) or the original question and chat history to generate a final
    response.

    Example:
        .. code-block:: python

            from langchain.chains import (
                StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
            )
            from langchain_core.prompts import PromptTemplate
            from langchain_community.llms import OpenAI

            combine_docs_chain = StuffDocumentsChain(...)
            vectorstore = ...
            retriever = vectorstore.as_retriever()

            # This controls how the standalone question is generated.
            # Should take `chat_history` and `question` as input variables.
            template = (
                "Combine the chat history and follow up question into "
                "a standalone question. Chat History: {chat_history}"
                "Follow up question: {question}"
            )
            prompt = PromptTemplate.from_template(template)
            llm = OpenAI()
            question_generator_chain = LLMChain(llm=llm, prompt=prompt)
            chain = ConversationalRetrievalChain(
                combine_docs_chain=combine_docs_chain,
                retriever=retriever,
                question_generator=question_generator_chain,
            )
    """
    retriever: BaseRetriever
    'Retriever to use to fetch documents.'
    max_tokens_limit: Optional[int] = None
    'If set, enforces that the documents returned are less than this limit.\n    This is only enforced if `combine_docs_chain` is of type StuffDocumentsChain.'

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)
        if self.max_tokens_limit and isinstance(self.combine_docs_chain, StuffDocumentsChain):
            tokens = [self.combine_docs_chain.llm_chain._get_num_tokens(doc.page_content) for doc in docs]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]
        return docs[:num_docs]

    def _get_docs(self, question: str, inputs: Dict[str, Any], *, run_manager: CallbackManagerForChainRun) -> List[Document]:
        """Get docs."""
        docs = self.retriever.get_relevant_documents(question, callbacks=run_manager.get_child())
        return self._reduce_tokens_below_limit(docs)

    async def _aget_docs(self, question: str, inputs: Dict[str, Any], *, run_manager: AsyncCallbackManagerForChainRun) -> List[Document]:
        """Get docs."""
        docs = await self.retriever.aget_relevant_documents(question, callbacks=run_manager.get_child())
        return self._reduce_tokens_below_limit(docs)

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, retriever: BaseRetriever, condense_question_prompt: BasePromptTemplate=CONDENSE_QUESTION_PROMPT, chain_type: str='stuff', verbose: bool=False, condense_question_llm: Optional[BaseLanguageModel]=None, combine_docs_chain_kwargs: Optional[Dict]=None, callbacks: Callbacks=None, **kwargs: Any) -> BaseConversationalRetrievalChain:
        """Convenience method to load chain from LLM and retriever.

        This provides some logic to create the `question_generator` chain
        as well as the combine_docs_chain.

        Args:
            llm: The default language model to use at every part of this chain
                (eg in both the question generation and the answering)
            retriever: The retriever to use to fetch relevant documents from.
            condense_question_prompt: The prompt to use to condense the chat history
                and new question into a standalone question.
            chain_type: The chain type to use to create the combine_docs_chain, will
                be sent to `load_qa_chain`.
            verbose: Verbosity flag for logging to stdout.
            condense_question_llm: The language model to use for condensing the chat
                history and new question into a standalone question. If none is
                provided, will default to `llm`.
            combine_docs_chain_kwargs: Parameters to pass as kwargs to `load_qa_chain`
                when constructing the combine_docs_chain.
            callbacks: Callbacks to pass to all subchains.
            **kwargs: Additional parameters to pass when initializing
                ConversationalRetrievalChain
        """
        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {}
        doc_chain = load_qa_chain(llm, chain_type=chain_type, verbose=verbose, callbacks=callbacks, **combine_docs_chain_kwargs)
        _llm = condense_question_llm or llm
        condense_question_chain = LLMChain(llm=_llm, prompt=condense_question_prompt, verbose=verbose, callbacks=callbacks)
        return cls(retriever=retriever, combine_docs_chain=doc_chain, question_generator=condense_question_chain, callbacks=callbacks, **kwargs)