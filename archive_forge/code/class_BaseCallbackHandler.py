from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
from tenacity import RetryCallState
class BaseCallbackHandler(LLMManagerMixin, ChainManagerMixin, ToolManagerMixin, RetrieverManagerMixin, CallbackManagerMixin, RunManagerMixin):
    """Base callback handler that handles callbacks from LangChain."""
    raise_error: bool = False
    run_inline: bool = False

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False

    @property
    def ignore_retry(self) -> bool:
        """Whether to ignore retry callbacks."""
        return False

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    @property
    def ignore_retriever(self) -> bool:
        """Whether to ignore retriever callbacks."""
        return False

    @property
    def ignore_chat_model(self) -> bool:
        """Whether to ignore chat model callbacks."""
        return False