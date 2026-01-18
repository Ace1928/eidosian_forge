import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple
import aiohttp
import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def _sync_request_embed(self, model: str, batch_texts: List[str]) -> List[List[float]]:
    response = requests.post(**self._kwargs_post_request(model=model, texts=batch_texts))
    if response.status_code != 200:
        raise Exception(f'Infinity returned an unexpected response with status {response.status_code}: {response.text}')
    return [e['embedding'] for e in response.json()['data']]