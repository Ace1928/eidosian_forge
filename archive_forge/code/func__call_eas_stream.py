import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, cast
import requests
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
def _call_eas_stream(self, query_body: dict) -> Any:
    """Generate text from the eas service."""
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'Authorization': f'{self.eas_service_token}'}
    response = requests.post(self.eas_service_url, headers=headers, json=query_body, timeout=self.timeout)
    if response.status_code != 200:
        raise Exception(f'Request failed with status code {response.status_code} and message {response.text}')
    return response