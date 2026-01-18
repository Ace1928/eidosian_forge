from __future__ import annotations
import json
import logging
from typing import (
import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from tenacity import (
def _get_embeddings(self, texts: List[str], batch_size: Optional[int]=None, input_type: Optional[str]=None) -> List[List[float]]:
    embeddings: List[List[float]] = []
    if batch_size is None:
        batch_size = self.batch_size
    if self.show_progress_bar:
        try:
            from tqdm.auto import tqdm
        except ImportError as e:
            raise ImportError('Must have tqdm installed if `show_progress_bar` is set to True. Please install with `pip install tqdm`.') from e
        _iter = tqdm(range(0, len(texts), batch_size))
    else:
        _iter = range(0, len(texts), batch_size)
    if input_type and input_type not in ['query', 'document']:
        raise ValueError(f"input_type {input_type} is invalid. Options: None, 'query', 'document'.")
    for i in _iter:
        response = embed_with_retry(self, **self._invocation_params(input=texts[i:i + batch_size], input_type=input_type))
        embeddings.extend((r['embedding'] for r in response['data']))
    return embeddings