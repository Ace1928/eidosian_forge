from __future__ import annotations
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
def _get_edenai(self, url: str) -> requests.Response:
    headers = {'accept': 'application/json', 'authorization': f'Bearer {self.edenai_api_key}', 'User-Agent': self.get_user_agent()}
    response = requests.get(url, headers=headers)
    self._raise_on_error(response)
    return response