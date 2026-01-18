import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
class _DatabricksClientBase(BaseModel, ABC):
    """A base JSON API client that talks to Databricks."""
    api_url: str
    api_token: str

    def request(self, method: str, url: str, request: Any) -> Any:
        headers = {'Authorization': f'Bearer {self.api_token}'}
        response = requests.request(method=method, url=url, headers=headers, json=request)
        if not response.ok:
            raise ValueError(f'HTTP {response.status_code} error: {response.text}')
        return response.json()

    def _get(self, url: str) -> Any:
        return self.request('GET', url, None)

    def _post(self, url: str, request: Any) -> Any:
        return self.request('POST', url, request)

    @abstractmethod
    def post(self, request: Any, transform_output_fn: Optional[Callable[..., str]]=None) -> Any:
        ...

    @property
    def llm(self) -> bool:
        return False