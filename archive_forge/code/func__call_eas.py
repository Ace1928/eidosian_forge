import json
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
def _call_eas(self, prompt: str='', params: Dict={}) -> Any:
    """Generate text from the eas service."""
    headers = {'Content-Type': 'application/json', 'Authorization': f'{self.eas_service_token}'}
    if self.version == '1.0':
        body = {'input_ids': f'{prompt}'}
    else:
        body = {'prompt': f'{prompt}'}
    for key, value in params.items():
        body[key] = value
    response = requests.post(self.eas_service_url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f'Request failed with status code {response.status_code} and message {response.text}')
    try:
        return json.loads(response.text)
    except Exception as e:
        if isinstance(e, json.decoder.JSONDecodeError):
            return response.text
        raise e