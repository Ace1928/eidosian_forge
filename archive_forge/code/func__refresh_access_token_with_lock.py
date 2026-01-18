import asyncio
import logging
import threading
from typing import Dict, List, Optional
import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import get_from_dict_or_env
def _refresh_access_token_with_lock(self) -> None:
    with self._lock:
        logger.debug('Refreshing access token')
        base_url: str = f'{self.ernie_api_base}/oauth/2.0/token'
        resp = requests.post(base_url, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}, params={'grant_type': 'client_credentials', 'client_id': self.ernie_client_id, 'client_secret': self.ernie_client_secret})
        self.access_token = str(resp.json().get('access_token'))