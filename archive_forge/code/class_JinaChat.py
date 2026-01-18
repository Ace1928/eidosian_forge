from __future__ import annotations
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
from tenacity import (
class JinaChat(BaseChatModel):
    """`Jina AI` Chat models API.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``JINACHAT_API_KEY`` set to your API key, which you
    can generate at https://chat.jina.ai/api.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import JinaChat
            chat = JinaChat()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {'jinachat_api_key': 'JINACHAT_API_KEY'}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False
    client: Any
    temperature: float = 0.7
    'What sampling temperature to use.'
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    'Holds any model parameters valid for `create` call not explicitly specified.'
    jinachat_api_key: Optional[SecretStr] = None
    'Base URL path for API requests, \n    leave blank if not using a proxy or service emulator.'
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    'Timeout for requests to JinaChat completion API. Default is 600 seconds.'
    max_retries: int = 6
    'Maximum number of retries to make when generating.'
    streaming: bool = False
    'Whether to stream the results or not.'
    max_tokens: Optional[int] = None
    'Maximum number of tokens to generate.'

    class Config:
        """Configuration for this pydantic object."""
        allow_population_by_field_name = True

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
        values['jinachat_api_key'] = convert_to_secret_str(get_from_dict_or_env(values, 'jinachat_api_key', 'JINACHAT_API_KEY'))
        try:
            import openai
        except ImportError:
            raise ValueError('Could not import openai python package. Please install it with `pip install openai`.')
        try:
            values['client'] = openai.ChatCompletion
        except AttributeError:
            raise ValueError('`openai` has no `ChatCompletion` attribute, this is likely due to an old version of the openai package. Try upgrading it with `pip install --upgrade openai`.')
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling JinaChat API."""
        return {'request_timeout': self.request_timeout, 'max_tokens': self.max_tokens, 'stream': self.streaming, 'temperature': self.temperature, **self.model_kwargs}

    def _create_retry_decorator(self) -> Callable[[Any], Any]:
        import openai
        min_seconds = 1
        max_seconds = 60
        return retry(reraise=True, stop=stop_after_attempt(self.max_retries), wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds), retry=retry_if_exception_type(openai.error.Timeout) | retry_if_exception_type(openai.error.APIError) | retry_if_exception_type(openai.error.APIConnectionError) | retry_if_exception_type(openai.error.RateLimitError) | retry_if_exception_type(openai.error.ServiceUnavailableError), before_sleep=before_sleep_log(logger, logging.WARNING))

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.create(**kwargs)
        return _completion_with_retry(**kwargs)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                continue
            token_usage = output['token_usage']
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {'token_usage': overall_token_usage}

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, 'stream': True}
        default_chunk_class = AIMessageChunk
        for chunk in self.completion_with_retry(messages=message_dicts, **params):
            delta = chunk['choices'][0]['delta']
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
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

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response['choices']:
            message = _convert_dict_to_message(res['message'])
            gen = ChatGeneration(message=message)
            generations.append(gen)
        llm_output = {'token_usage': response['usage']}
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, 'stream': True}
        default_chunk_class = AIMessageChunk
        async for chunk in await acompletion_with_retry(self, messages=message_dicts, **params):
            delta = chunk['choices'][0]['delta']
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
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
        jinachat_creds: Dict[str, Any] = {'api_key': self.jinachat_api_key and self.jinachat_api_key.get_secret_value(), 'api_base': 'https://api.chat.jina.ai/v1', 'model': 'jinachat'}
        return {**jinachat_creds, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return 'jinachat'