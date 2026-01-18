from typing import Any, Dict, List, Literal, Optional
import aiohttp
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
class YouHitMetadata(BaseModel):
    """Metadata on a single hit from you.com"""
    title: str = Field(description='The title of the result')
    url: str = Field(description='The url of the result')
    thumbnail_url: str = Field(description='Thumbnail associated with the result')
    description: str = Field(description='Details about the result')