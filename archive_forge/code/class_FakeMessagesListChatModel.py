import asyncio
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
class FakeMessagesListChatModel(BaseChatModel):
    """Fake ChatModel for testing purposes."""
    responses: List[BaseMessage]
    sleep: Optional[float] = None
    i: int = 0

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return 'fake-messages-list-chat-model'