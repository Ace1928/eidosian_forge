import json
import logging
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_pydantic_field_names
class LlamaEdgeChatService(BaseChatModel):
    """Chat with LLMs via `llama-api-server`

    For the information about `llama-api-server`, visit https://github.com/second-state/LlamaEdge
    """
    request_timeout: int = 60
    'request timeout for chat http requests'
    service_url: Optional[str] = None
    'URL of WasmChat service'
    model: str = 'NA'
    'model name, default is `NA`.'
    streaming: bool = False
    'Whether to stream the results or not.'

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

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(messages=messages, stop=stop, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)
        res = self._chat(messages, **kwargs)
        if res.status_code != 200:
            raise ValueError(f'Error code: {res.status_code}, reason: {res.reason}')
        response = res.json()
        return self._create_chat_result(response)

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        res = self._chat(messages, **kwargs)
        default_chunk_class = AIMessageChunk
        substring = '"object":"chat.completion.chunk"}'
        for line in res.iter_lines():
            chunks = []
            if line:
                json_string = line.decode('utf-8')
                positions = [m.start() for m in re.finditer(substring, json_string)]
                positions = [-1 * len(substring)] + positions
                for i in range(len(positions) - 1):
                    chunk = json.loads(json_string[positions[i] + len(substring):positions[i + 1] + len(substring)])
                    chunks.append(chunk)
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    chunk = chunk.dict()
                if len(chunk['choices']) == 0:
                    continue
                choice = chunk['choices'][0]
                chunk = _convert_delta_to_message_chunk(choice['delta'], default_chunk_class)
                if choice.get('finish_reason') is not None and choice.get('finish_reason') == 'stop':
                    break
                finish_reason = choice.get('finish_reason')
                generation_info = dict(finish_reason=finish_reason) if finish_reason is not None else None
                default_chunk_class = chunk.__class__
                cg_chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
                if run_manager:
                    run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
                yield cg_chunk

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> requests.Response:
        if self.service_url is None:
            res = requests.models.Response()
            res.status_code = 503
            res.reason = 'The IP address or port of the chat service is incorrect.'
            return res
        service_url = f'{self.service_url}/v1/chat/completions'
        if self.streaming:
            payload = {'model': self.model, 'messages': [_convert_message_to_dict(m) for m in messages], 'stream': self.streaming}
        else:
            payload = {'model': self.model, 'messages': [_convert_message_to_dict(m) for m in messages]}
        res = requests.post(url=service_url, timeout=self.request_timeout, headers={'accept': 'application/json', 'Content-Type': 'application/json'}, data=json.dumps(payload))
        return res

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        message = _convert_dict_to_message(response['choices'][0].get('message'))
        generations = [ChatGeneration(message=message)]
        token_usage = response['usage']
        llm_output = {'token_usage': token_usage, 'model': self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return 'wasm-chat'