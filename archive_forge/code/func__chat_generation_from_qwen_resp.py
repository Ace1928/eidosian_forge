from __future__ import annotations
import asyncio
import functools
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.openai_tools import (
from langchain_core.outputs import (
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
from langchain_community.llms.tongyi import (
@staticmethod
def _chat_generation_from_qwen_resp(resp: Any, is_chunk: bool=False, is_last_chunk: bool=True) -> Dict[str, Any]:
    choice = resp['output']['choices'][0]
    message = convert_dict_to_message(choice['message'], is_chunk=is_chunk)
    if is_last_chunk:
        return dict(message=message, generation_info=dict(finish_reason=choice['finish_reason'], request_id=resp['request_id'], token_usage=dict(resp['usage'])))
    else:
        return dict(message=message)