from __future__ import annotations
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.adapters.openai import (
from langchain_community.chat_models.openai import _convert_delta_to_message_chunk
class GPTRouter(BaseChatModel):
    """GPTRouter by Writesonic Inc.

    For more information, see https://gpt-router.writesonic.com/docs
    """
    client: Any = Field(default=None, exclude=True)
    models_priority_list: List[GPTRouterModel] = Field(min_items=1)
    gpt_router_api_base: str = Field(default=None)
    'WriteSonic GPTRouter custom endpoint'
    gpt_router_api_key: Optional[SecretStr] = None
    'WriteSonic GPTRouter API Key'
    temperature: float = 0.7
    'What sampling temperature to use.'
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    'Holds any model parameters valid for `create` call not explicitly specified.'
    max_retries: int = 4
    'Maximum number of retries to make when generating.'
    streaming: bool = False
    'Whether to stream the results or not.'
    n: int = 1
    'Number of chat completions to generate for each prompt.'
    max_tokens: int = 256

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        values['gpt_router_api_base'] = get_from_dict_or_env(values, 'gpt_router_api_base', 'GPT_ROUTER_API_BASE', DEFAULT_API_BASE_URL)
        values['gpt_router_api_key'] = convert_to_secret_str(get_from_dict_or_env(values, 'gpt_router_api_key', 'GPT_ROUTER_API_KEY'))
        try:
            from gpt_router.client import GPTRouterClient
        except ImportError:
            raise GPTRouterException('Could not import GPTRouter python package. Please install it with `pip install GPTRouter`.')
        gpt_router_client = GPTRouterClient(values['gpt_router_api_base'], values['gpt_router_api_key'].get_secret_value())
        values['client'] = gpt_router_client
        return values

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {'gpt_router_api_key': 'GPT_ROUTER_API_KEY'}

    @property
    def lc_serializable(self) -> bool:
        return True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return 'gpt-router-chat'

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{'models_priority_list': self.models_priority_list}, **self._default_params}

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling GPTRouter API."""
        return {'max_tokens': self.max_tokens, 'stream': self.streaming, 'n': self.n, 'temperature': self.temperature, **self.model_kwargs}

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, stream: Optional[bool]=None, **kwargs: Any) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, 'stream': False}
        response = completion_with_retry(self, messages=message_dicts, models_priority_list=self.models_priority_list, run_manager=run_manager, **params)
        return self._create_chat_result(response)

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, stream: Optional[bool]=None, **kwargs: Any) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return await agenerate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, 'stream': False}
        response = await acompletion_with_retry(self, messages=message_dicts, models_priority_list=self.models_priority_list, run_manager=run_manager, **params)
        return self._create_chat_result(response)

    def _create_chat_generation_chunk(self, data: Mapping[str, Any], default_chunk_class: Type[BaseMessageChunk]) -> Tuple[ChatGenerationChunk, Type[BaseMessageChunk]]:
        chunk = _convert_delta_to_message_chunk({'content': data.get('text', '')}, default_chunk_class)
        finish_reason = data.get('finish_reason')
        generation_info = dict(finish_reason=finish_reason) if finish_reason is not None else None
        default_chunk_class = chunk.__class__
        gen_chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
        return (gen_chunk, default_chunk_class)

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, 'stream': True}
        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        generator_response = completion_with_retry(self, messages=message_dicts, models_priority_list=self.models_priority_list, run_manager=run_manager, **params)
        for chunk in generator_response:
            if chunk.event != 'update':
                continue
            chunk, default_chunk_class = self._create_chat_generation_chunk(chunk.data, default_chunk_class)
            if run_manager:
                run_manager.on_llm_new_token(token=chunk.message.content, chunk=chunk.message)
            yield chunk

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, 'stream': True}
        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        generator_response = acompletion_with_retry(self, messages=message_dicts, models_priority_list=self.models_priority_list, run_manager=run_manager, **params)
        async for chunk in await generator_response:
            if chunk.event != 'update':
                continue
            chunk, default_chunk_class = self._create_chat_generation_chunk(chunk.data, default_chunk_class)
            if run_manager:
                await run_manager.on_llm_new_token(token=chunk.message.content, chunk=chunk.message)
            yield chunk

    def _create_message_dicts(self, messages: List[BaseMessage], stop: Optional[List[str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            if 'stop' in params:
                raise ValueError('`stop` found in both the input and default params.')
            params['stop'] = stop
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return (message_dicts, params)

    def _create_chat_result(self, response: GenerationResponse) -> ChatResult:
        generations = []
        for res in response.choices:
            message = convert_dict_to_message({'role': 'assistant', 'content': res.text})
            gen = ChatGeneration(message=message, generation_info=dict(finish_reason=res.finish_reason))
            generations.append(gen)
        llm_output = {'token_usage': response.meta, 'model': response.model}
        return ChatResult(generations=generations, llm_output=llm_output)