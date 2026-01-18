from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import Field, PrivateAttr
def _validate_uri(self) -> None:
    if self.target_uri == 'databricks':
        return
    allowed = ['http', 'https', 'databricks']
    if urlparse(self.target_uri).scheme not in allowed:
        raise ValueError(f'Invalid target URI: {self.target_uri}. The scheme must be one of {allowed}.')