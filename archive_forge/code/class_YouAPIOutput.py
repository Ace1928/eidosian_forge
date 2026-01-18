from typing import Any, Dict, List, Literal, Optional
import aiohttp
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
class YouAPIOutput(BaseModel):
    """Output from you.com API."""
    hits: List[YouHit] = Field(description='A list of dictionaries containing the results')