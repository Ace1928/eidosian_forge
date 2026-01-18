import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests import Request, Session
def _create_action_url(self, action_id: str) -> str:
    """Create a url for an action."""
    return self.zapier_nla_api_base + f'exposed/{action_id}/execute/'