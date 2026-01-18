from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_community.llms.openai import BaseOpenAI
from langchain_community.utils.openai import is_openai_v1
class VLLMOpenAI(BaseOpenAI):
    """vLLM OpenAI-compatible API client"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        params: Dict[str, Any] = {'model': self.model_name, **self._default_params, 'logit_bias': None}
        if not is_openai_v1():
            params.update({'api_key': self.openai_api_key, 'api_base': self.openai_api_base})
        return params

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'vllm-openai'