import asyncio
import re
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
class ParrotFakeChatModel(BaseChatModel):
    """A generic fake chat model that can be used to test the chat model interface.

    * Chat model should be usable in both sync and async tests
    """

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        """Top Level call"""
        return ChatResult(generations=[ChatGeneration(message=messages[-1])])

    @property
    def _llm_type(self) -> str:
        return 'parrot-fake-chat-model'