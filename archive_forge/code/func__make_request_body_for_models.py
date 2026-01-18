from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.retrievers import Document
def _make_request_body_for_models(self, prompt: str, **kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Make the request body for generate/retrieve models endpoint"""
    _model_kwargs = self.model_kwargs or {}
    _params = {**_model_kwargs, **kwargs}
    filters = [DALMFilter(**f) for f in _params.get('filters', [])]
    return dict(model_id=self.model_id, query=prompt, size=_params.get('size', 3), filters=filters, id=self.model_id)