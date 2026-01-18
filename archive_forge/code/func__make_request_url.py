from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.retrievers import Document
def _make_request_url(self, route: Union[ArceeRoute, str]) -> str:
    return f'{self.arcee_api_url}/{self.arcee_api_version}/{route}'