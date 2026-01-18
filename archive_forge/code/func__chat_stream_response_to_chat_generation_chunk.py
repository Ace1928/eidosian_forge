import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.ollama import OllamaEndpointNotFoundError, _OllamaCommon
def _chat_stream_response_to_chat_generation_chunk(stream_response: str) -> ChatGenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get('done') is True else None
    return ChatGenerationChunk(message=AIMessageChunk(content=parsed_response.get('message', {}).get('content', '')), generation_info=generation_info)