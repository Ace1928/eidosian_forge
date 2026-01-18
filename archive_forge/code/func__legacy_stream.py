import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.ollama import OllamaEndpointNotFoundError, _OllamaCommon
@deprecated('0.0.3', alternative='_stream')
def _legacy_stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
    prompt = self._format_messages_as_text(messages)
    for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
        if stream_resp:
            chunk = _stream_response_to_chat_generation_chunk(stream_resp)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, verbose=self.verbose)
            yield chunk