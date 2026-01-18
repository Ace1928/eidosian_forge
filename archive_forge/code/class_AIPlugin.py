from __future__ import annotations
import json
from typing import Optional, Type
import requests
import yaml
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
class AIPlugin(BaseModel):
    """AI Plugin Definition."""
    schema_version: str
    name_for_model: str
    name_for_human: str
    description_for_model: str
    description_for_human: str
    auth: Optional[dict] = None
    api: ApiConfig
    logo_url: Optional[str]
    contact_email: Optional[str]
    legal_info_url: Optional[str]

    @classmethod
    def from_url(cls, url: str) -> AIPlugin:
        """Instantiate AIPlugin from a URL."""
        response = requests.get(url).json()
        return cls(**response)