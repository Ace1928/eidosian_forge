import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional
import aiohttp
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.requests import Requests
def _handle_status(self, code: int, text: Any) -> None:
    if code >= 500:
        raise Exception(f'DeepInfra Server: Error {code}')
    elif code >= 400:
        raise ValueError(f'DeepInfra received an invalid payload: {text}')
    elif code != 200:
        raise Exception(f'DeepInfra returned an unexpected response with status {code}: {text}')