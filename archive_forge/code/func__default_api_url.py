from typing import Any, Dict, List, Optional
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, SecretStr
@property
def _default_api_url(self) -> str:
    return f'https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}'