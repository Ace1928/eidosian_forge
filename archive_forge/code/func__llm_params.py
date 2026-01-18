import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
@property
def _llm_params(self) -> Dict[str, Any]:
    params: Dict[str, Any] = {'temperature': self.temperature, 'n': self.n}
    if self.stop:
        params['stop'] = self.stop
    if self.max_tokens is not None:
        params['max_tokens'] = self.max_tokens
    return params