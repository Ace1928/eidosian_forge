from typing import (
from langchain_core.callbacks.manager import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_community.chat_models.litellm import (
class ChatLiteLLMRouter(ChatLiteLLM):
    """LiteLLM Router as LangChain Model."""
    router: Any

    def __init__(self, *, router: Any, **kwargs: Any) -> None:
        """Construct Chat LiteLLM Router."""
        super().__init__(**kwargs)
        self.router = router

    @property
    def _llm_type(self) -> str:
        return 'LiteLLMRouter'

    def _set_model_for_completion(self) -> None:
        self.model = self.router.model_list[0]['model_name']

    def _prepare_params_for_router(self, params: Any) -> None:
        params['model'] = self.model
        api_base_key_name = 'api_base'
        if api_base_key_name in params and params[api_base_key_name] is None:
            del params[api_base_key_name]
        params.setdefault('metadata', {})

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, stream: Optional[bool]=None, **kwargs: Any) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        self._set_model_for_completion()
        self._prepare_params_for_router(params)
        response = self.router.completion(messages=message_dicts, **params)
        return self._create_chat_result(response, **params)

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        default_chunk_class = AIMessageChunk
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, 'stream': True}
        self._set_model_for_completion()
        self._prepare_params_for_router(params)
        for chunk in self.router.completion(messages=message_dicts, **params):
            if len(chunk['choices']) == 0:
                continue
            delta = chunk['choices'][0]['delta']
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk, **params)
            yield cg_chunk

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        default_chunk_class = AIMessageChunk
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, 'stream': True}
        self._set_model_for_completion()
        self._prepare_params_for_router(params)
        async for chunk in await self.router.acompletion(messages=message_dicts, **params):
            if len(chunk['choices']) == 0:
                continue
            delta = chunk['choices'][0]['delta']
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk, **params)
            yield cg_chunk

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, stream: Optional[bool]=None, **kwargs: Any) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(messages=messages, stop=stop, run_manager=run_manager, **kwargs)
            return await agenerate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        self._set_model_for_completion()
        self._prepare_params_for_router(params)
        response = await self.router.acompletion(messages=message_dicts, **params)
        return self._create_chat_result(response, **params)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                continue
            token_usage = output['token_usage']
            if token_usage is not None:
                for k, v in token_usage.dict().items():
                    if k in overall_token_usage:
                        overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get('system_fingerprint')
        combined = {'token_usage': overall_token_usage, 'model_name': self.model_name}
        if system_fingerprint:
            combined['system_fingerprint'] = system_fingerprint
        return combined

    def _create_chat_result(self, response: Mapping[str, Any], **params: Any) -> ChatResult:
        from litellm.utils import Usage
        generations = []
        for res in response['choices']:
            message = _convert_dict_to_message(res['message'])
            gen = ChatGeneration(message=message, generation_info=dict(finish_reason=res.get('finish_reason')))
            generations.append(gen)
        token_usage = response.get('usage', Usage(prompt_tokens=0, total_tokens=0))
        llm_output = get_llm_output(token_usage, **params)
        return ChatResult(generations=generations, llm_output=llm_output)