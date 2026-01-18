from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Literal, Optional, Union
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra
from requests import Response
class TextRequestsWrapper(GenericRequestsWrapper):
    """Lightweight wrapper around requests library, with async support.

    The main purpose of this wrapper is to always return a text output."""
    response_content_type: Literal['text', 'json'] = 'text'