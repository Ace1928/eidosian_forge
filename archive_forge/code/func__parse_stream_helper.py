import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional
import aiohttp
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.requests import Requests
def _parse_stream_helper(line: bytes) -> Optional[str]:
    if line and line.startswith(b'data:'):
        if line.startswith(b'data: '):
            line = line[len(b'data: '):]
        else:
            line = line[len(b'data:'):]
        if line.strip() == b'[DONE]':
            return None
        else:
            return line.decode('utf-8')
    return None