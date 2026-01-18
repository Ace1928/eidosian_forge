from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
class Solar(SolarCommon, LLM):
    """Solar large language models.

    To use, you should have the environment variable
    ``SOLAR_API_KEY`` set with your API key.
    Referenced from https://console.upstage.ai/services/solar
    """

    class Config:
        allow_population_by_field_name = True

    def _call(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> str:
        request = self._invocation_params
        request['messages'] = [{'role': 'user', 'content': prompt}]
        request.update(kwargs)
        text = self._client.completion(request)
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text