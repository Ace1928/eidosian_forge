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
def _format_response_payload(self, output: bytes, stop_sequences: Optional[List[str]]) -> str:
    """Formats response"""
    try:
        text = json.loads(output)['response']
        if stop_sequences:
            text = enforce_stop_tokens(text, stop_sequences)
        return text
    except Exception as e:
        if isinstance(e, json.decoder.JSONDecodeError):
            return output.decode('utf-8')
        raise e