import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.ollama import OllamaEndpointNotFoundError, _OllamaCommon
def _create_chat_stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, **kwargs: Any) -> Iterator[str]:
    payload = {'model': self.model, 'messages': self._convert_messages_to_ollama_messages(messages)}
    yield from self._create_stream(payload=payload, stop=stop, api_url=f'{self.base_url}/api/chat', **kwargs)