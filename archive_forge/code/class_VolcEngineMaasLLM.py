from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
class VolcEngineMaasLLM(LLM, VolcEngineMaasBase):
    """volc engine maas hosts a plethora of models.
    You can utilize these models through this class.

    To use, you should have the ``volcengine`` python package installed.
    and set access key and secret key by environment variable or direct pass those to
    this class.
    access key, secret key are required parameters which you could get help
    https://www.volcengine.com/docs/6291/65568

    In order to use them, it is necessary to install the 'volcengine' Python package.
    The access key and secret key must be set either via environment variables or
    passed directly to this class.
    access key and secret key are mandatory parameters for which assistance can be
    sought at https://www.volcengine.com/docs/6291/65568.

    Example:
        .. code-block:: python

            from langchain_community.llms import VolcEngineMaasLLM
            model = VolcEngineMaasLLM(model="skylark-lite-public",
                                          volc_engine_maas_ak="your_ak",
                                          volc_engine_maas_sk="your_sk")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'volc-engine-maas-llm'

    def _convert_prompt_msg_params(self, prompt: str, **kwargs: Any) -> dict:
        model_req = {'model': {'name': self.model}}
        if self.model_version is not None:
            model_req['model']['version'] = self.model_version
        return {**model_req, 'messages': [{'role': 'user', 'content': prompt}], 'parameters': {**self._default_params, **kwargs}}

    def _call(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> str:
        if self.streaming:
            completion = ''
            for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion
        params = self._convert_prompt_msg_params(prompt, **kwargs)
        response = self.client.chat(params)
        return response.get('choice', {}).get('message', {}).get('content', '')

    def _stream(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[GenerationChunk]:
        params = self._convert_prompt_msg_params(prompt, **kwargs)
        for res in self.client.stream_chat(params):
            if res:
                chunk = GenerationChunk(text=res.get('choice', {}).get('message', {}).get('content', ''))
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk