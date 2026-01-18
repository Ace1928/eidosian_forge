from __future__ import annotations
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import (
from tenacity import (
class ChatYuan2(BaseChatModel):
    """`Yuan2.0` Chat models API.

    To use, you should have the ``openai-python`` package installed, if package
    not installed, using ```pip install openai``` to install it. The
    environment variable ``YUAN2_API_KEY`` set to your API key, if not set,
    everyone can access apis.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatYuan2

            chat = ChatYuan2()
    """
    client: Any
    async_client: Any = Field(default=None, exclude=True)
    model_name: str = Field(default='yuan2', alias='model')
    'Model name to use.'
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    'Holds any model parameters valid for `create` call not explicitly specified.'
    yuan2_api_key: Optional[str] = Field(default='EMPTY', alias='api_key')
    'Automatically inferred from env var `YUAN2_API_KEY` if not provided.'
    yuan2_api_base: Optional[str] = Field(default='http://127.0.0.1:8000/v1', alias='base_url')
    'Base URL path for API requests, an OpenAI compatible API server.'
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    'Timeout for requests to yuan2 completion API. Default is 600 seconds.'
    max_retries: int = 6
    'Maximum number of retries to make when generating.'
    streaming: bool = False
    'Whether to stream the results or not.'
    max_tokens: Optional[int] = None
    'Maximum number of tokens to generate.'
    temperature: float = 1.0
    'What sampling temperature to use.'
    top_p: Optional[float] = 0.9
    'The top-p value to use for sampling.'
    stop: Optional[List[str]] = ['<eod>']
    'A list of strings to stop generation when encountered.'
    repeat_last_n: Optional[int] = 64
    'Last n tokens to penalize'
    repeat_penalty: Optional[float] = 1.18
    'The penalty to apply to repeated tokens.'

    class Config:
        """Configuration for this pydantic object."""
        allow_population_by_field_name = True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {'yuan2_api_key': 'YUAN2_API_KEY'}

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        if self.yuan2_api_base:
            attributes['yuan2_api_base'] = self.yuan2_api_base
        if self.yuan2_api_key:
            attributes['yuan2_api_key'] = self.yuan2_api_key
        return attributes

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get('model_kwargs', {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f'Found {field_name} supplied twice.')
            if field_name not in all_required_field_names:
                logger.warning(f'WARNING! {field_name} is not default parameter.\n                    {field_name} was transferred to model_kwargs.\n                    Please confirm that {field_name} is what you intended.')
                extra[field_name] = values.pop(field_name)
        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(f'Parameters {invalid_model_kwargs} should be specified explicitly. Instead they were passed in as part of `model_kwargs` parameter.')
        values['model_kwargs'] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values['yuan2_api_key'] = get_from_dict_or_env(values, 'yuan2_api_key', 'YUAN2_API_KEY')
        try:
            import openai
        except ImportError:
            raise ValueError('Could not import openai python package. Please install it with `pip install openai`.')
        client_params = {'api_key': values['yuan2_api_key'], 'base_url': values['yuan2_api_base'], 'timeout': values['request_timeout'], 'max_retries': values['max_retries']}
        if not values.get('client'):
            values['client'] = openai.OpenAI(**client_params).chat.completions
        if not values.get('async_client'):
            values['async_client'] = openai.AsyncOpenAI(**client_params).chat.completions
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling yuan2 API."""
        params = {'model': self.model_name, 'stream': self.streaming, 'temperature': self.temperature, 'top_p': self.top_p, **self.model_kwargs}
        if self.max_tokens is not None:
            params['max_tokens'] = self.max_tokens
        if self.request_timeout is not None:
            params['request_timeout'] = self.request_timeout
        return params

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.create(**kwargs)
        return _completion_with_retry(**kwargs)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        logger.debug(f'type(llm_outputs): {type(llm_outputs)}; llm_outputs: {llm_outputs}')
        for output in llm_outputs:
            if output is None:
                continue
            token_usage = output['token_usage']
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {'token_usage': overall_token_usage, 'model_name': self.model_name}

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, 'stream': True}
        default_chunk_class = AIMessageChunk
        for chunk in self.completion_with_retry(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk['choices']) == 0:
                continue
            choice = chunk['choices'][0]
            chunk = _convert_delta_to_message_chunk(choice['delta'], default_chunk_class)
            finish_reason = choice.get('finish_reason')
            generation_info = dict(finish_reason=finish_reason) if finish_reason is not None else None
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
            yield cg_chunk

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(messages=messages, stop=stop, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.completion_with_retry(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _create_message_dicts(self, messages: List[BaseMessage], stop: Optional[List[str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = dict(self._invocation_params)
        if stop is not None:
            if 'stop' in params:
                raise ValueError('`stop` found in both the input and default params.')
            params['stop'] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return (message_dicts, params)

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        generations = []
        logger.debug(f'type(response): {type(response)}; response: {response}')
        if not isinstance(response, dict):
            response = response.dict()
        for res in response['choices']:
            message = _convert_dict_to_message(res['message'])
            generation_info = dict(finish_reason=res['finish_reason'])
            if 'logprobs' in res:
                generation_info['logprobs'] = res['logprobs']
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)
        llm_output = {'token_usage': response.get('usage', {}), 'model_name': self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, 'stream': True}
        default_chunk_class = AIMessageChunk
        async for chunk in await acompletion_with_retry(self, messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk['choices']) == 0:
                continue
            choice = chunk['choices'][0]
            chunk = _convert_delta_to_message_chunk(choice['delta'], default_chunk_class)
            finish_reason = choice.get('finish_reason')
            generation_info = dict(finish_reason=finish_reason) if finish_reason is not None else None
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
            yield cg_chunk

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(messages=messages, stop=stop, run_manager=run_manager, **kwargs)
            return await agenerate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await acompletion_with_retry(self, messages=message_dicts, **params)
        return self._create_chat_result(response)

    @property
    def _invocation_params(self) -> Mapping[str, Any]:
        """Get the parameters used to invoke the model."""
        yuan2_creds: Dict[str, Any] = {'model': self.model_name}
        return {**yuan2_creds, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return 'chat-yuan2'