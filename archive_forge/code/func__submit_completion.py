import json
import logging
import os
import re
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.transform import BaseOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
def _submit_completion(self, messages: List[Dict]) -> _KdtSqlResponse:
    """Submit a /chat/completions request to Kinetica."""
    request = dict(messages=messages)
    request_json = json.dumps(request)
    response_raw = self.kdbc._GPUdb__submit_request_json('/chat/completions', request_json)
    response_json = json.loads(response_raw)
    status = response_json['status']
    if status != 'OK':
        message = response_json['message']
        match_resp = re.compile('response:({.*})')
        result = match_resp.search(message)
        if result is not None:
            response = result.group(1)
            response_json = json.loads(response)
            message = response_json['message']
        raise ValueError(message)
    data = response_json['data']
    response = _KdtCompletionResponse.parse_obj(data)
    if response.status != 'OK':
        raise ValueError('SQL Generation failed')
    return response.data