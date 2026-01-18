import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests import Request, Session
def _create_action_request(self, action_id: str, instructions: str, params: Optional[Dict]=None, preview_only=False) -> Request:
    data = self._create_action_payload(instructions, params, preview_only)
    return Request('POST', self._create_action_url(action_id), json=data)