from __future__ import annotations
import asyncio
import inspect
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import (
from langchain_core._api import deprecated
from langchain_core.caches import BaseCache
from langchain_core.callbacks import (
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import BaseLanguageModel, LanguageModelInput
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.runnables.config import ensure_config, run_in_executor
from langchain_core.tracers.log_stream import LogStreamCallbackHandler
class SimpleChatModel(BaseChatModel):
    """A simplified implementation for a chat model to inherit from."""

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        output_str = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @abstractmethod
    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> str:
        """Simpler interface."""

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        return await run_in_executor(None, self._generate, messages, stop=stop, run_manager=run_manager.get_sync() if run_manager else None, **kwargs)