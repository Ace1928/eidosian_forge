import asyncio
import re
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
class FakeChatModel(SimpleChatModel):
    """Fake Chat Model wrapper for testing purposes."""

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> str:
        return 'fake response'

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        output_str = 'fake response'
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return 'fake-chat-model'

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {'key': 'fake'}