from __future__ import annotations
import asyncio
import functools
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
class Tongyi(BaseLLM):
    """Tongyi Qwen large language models.

    To use, you should have the ``dashscope`` python package installed, and the
    environment variable ``DASHSCOPE_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.llms import Tongyi
            tongyi = tongyi()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {'dashscope_api_key': 'DASHSCOPE_API_KEY'}
    client: Any
    model_name: str = 'qwen-plus'
    'Model name to use.'
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    top_p: float = 0.8
    'Total probability mass of tokens to consider at each step.'
    dashscope_api_key: Optional[str] = None
    'Dashscope api key provide by Alibaba Cloud.'
    streaming: bool = False
    'Whether to stream the results or not.'
    max_retries: int = 10
    'Maximum number of retries to make when generating.'

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'tongyi'

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values['dashscope_api_key'] = get_from_dict_or_env(values, 'dashscope_api_key', 'DASHSCOPE_API_KEY')
        try:
            import dashscope
        except ImportError:
            raise ImportError('Could not import dashscope python package. Please install it with `pip install dashscope`.')
        try:
            values['client'] = dashscope.Generation
        except AttributeError:
            raise ValueError('`dashscope` has no `Generation` attribute, this is likely due to an old version of the dashscope package. Try upgrading it with `pip install --upgrade dashscope`.')
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Tongyi Qwen API."""
        normal_params = {'model': self.model_name, 'top_p': self.top_p, 'api_key': self.dashscope_api_key}
        return {**normal_params, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'model_name': self.model_name, **super()._identifying_params}

    def _generate(self, prompts: List[str], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> LLMResult:
        generations = []
        if self.streaming:
            if len(prompts) > 1:
                raise ValueError('Cannot stream results with multiple prompts.')
            generation: Optional[GenerationChunk] = None
            for chunk in self._stream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            generations.append([self._chunk_to_generation(generation)])
        else:
            params: Dict[str, Any] = self._invocation_params(stop=stop, **kwargs)
            for prompt in prompts:
                completion = generate_with_retry(self, prompt=prompt, **params)
                generations.append([Generation(**self._generation_from_qwen_resp(completion))])
        return LLMResult(generations=generations, llm_output={'model_name': self.model_name})

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> LLMResult:
        generations = []
        if self.streaming:
            if len(prompts) > 1:
                raise ValueError('Cannot stream results with multiple prompts.')
            generation: Optional[GenerationChunk] = None
            async for chunk in self._astream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            generations.append([self._chunk_to_generation(generation)])
        else:
            params: Dict[str, Any] = self._invocation_params(stop=stop, **kwargs)
            for prompt in prompts:
                completion = await asyncio.get_running_loop().run_in_executor(None, functools.partial(generate_with_retry, **{'llm': self, 'prompt': prompt, **params}))
                generations.append([Generation(**self._generation_from_qwen_resp(completion))])
        return LLMResult(generations=generations, llm_output={'model_name': self.model_name})

    def _stream(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[GenerationChunk]:
        params: Dict[str, Any] = self._invocation_params(stop=stop, stream=True, **kwargs)
        for stream_resp, is_last_chunk in generate_with_last_element_mark(stream_generate_with_retry(self, prompt=prompt, **params)):
            chunk = GenerationChunk(**self._generation_from_qwen_resp(stream_resp, is_last_chunk))
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, verbose=self.verbose)
            yield chunk

    async def _astream(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
        params: Dict[str, Any] = self._invocation_params(stop=stop, stream=True, **kwargs)
        async for stream_resp, is_last_chunk in agenerate_with_last_element_mark(astream_generate_with_retry(self, prompt=prompt, **params)):
            chunk = GenerationChunk(**self._generation_from_qwen_resp(stream_resp, is_last_chunk))
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk, verbose=self.verbose)
            yield chunk

    def _invocation_params(self, stop: Any, **kwargs: Any) -> Dict[str, Any]:
        params = {**self._default_params, **kwargs}
        if stop is not None:
            params['stop'] = stop
        if params.get('stream'):
            params['incremental_output'] = True
        return params

    @staticmethod
    def _generation_from_qwen_resp(resp: Any, is_last_chunk: bool=True) -> Dict[str, Any]:
        if is_last_chunk:
            return dict(text=resp['output']['text'], generation_info=dict(finish_reason=resp['output']['finish_reason'], request_id=resp['request_id'], token_usage=dict(resp['usage'])))
        else:
            return dict(text=resp['output']['text'])

    @staticmethod
    def _chunk_to_generation(chunk: GenerationChunk) -> Generation:
        return Generation(text=chunk.text, generation_info=chunk.generation_info)