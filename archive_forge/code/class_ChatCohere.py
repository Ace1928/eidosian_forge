from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.cohere import BaseCohere
@deprecated(since='0.0.30', removal='0.2.0', alternative_import='langchain_cohere.ChatCohere')
class ChatCohere(BaseChatModel, BaseCohere):
    """`Cohere` chat large language models.

    To use, you should have the ``cohere`` python package installed, and the
    environment variable ``COHERE_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatCohere
            from langchain_core.messages import HumanMessage

            chat = ChatCohere(model="command", max_tokens=256, temperature=0.75)

            messages = [HumanMessage(content="knock knock")]
            chat.invoke(messages)
    """

    class Config:
        """Configuration for this pydantic object."""
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return 'cohere-chat'

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        return {'temperature': self.temperature}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{'model': self.model}, **self._default_params}

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)
        if hasattr(self.client, 'chat_stream'):
            stream = self.client.chat_stream(**request)
        else:
            stream = self.client.chat(**request, stream=True)
        for data in stream:
            if data.event_type == 'text-generation':
                delta = data.text
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)
        if hasattr(self.async_client, 'chat_stream'):
            stream = await self.async_client.chat_stream(**request)
        else:
            stream = await self.async_client.chat(**request, stream=True)
        async for data in stream:
            if data.event_type == 'text-generation':
                delta = data.text
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    await run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk

    def _get_generation_info(self, response: Any) -> Dict[str, Any]:
        """Get the generation info from cohere API response."""
        return {'documents': response.documents, 'citations': response.citations, 'search_results': response.search_results, 'search_queries': response.search_queries, 'token_count': response.token_count}

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)
        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)
        response = self.client.chat(**request)
        message = AIMessage(content=response.text)
        generation_info = None
        if hasattr(response, 'documents'):
            generation_info = self._get_generation_info(response)
        return ChatResult(generations=[ChatGeneration(message=message, generation_info=generation_info)])

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return await agenerate_from_stream(stream_iter)
        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)
        response = self.client.chat(**request)
        message = AIMessage(content=response.text)
        generation_info = None
        if hasattr(response, 'documents'):
            generation_info = self._get_generation_info(response)
        return ChatResult(generations=[ChatGeneration(message=message, generation_info=generation_info)])

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        return len(self.client.tokenize(text=text).tokens)