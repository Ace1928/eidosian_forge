import asyncio
import json
from threading import Lock
from typing import List, Union
from enum import Enum
import base64
from fastapi import APIRouter, Request, status, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
import tiktoken
from utils.rwkv import *
from utils.log import quick_log
import global_var
class ChatCompletionBody(ModelConfigBody):
    messages: Union[List[Message], None]
    model: Union[str, None] = 'rwkv'
    stream: bool = False
    stop: Union[str, List[str], None] = default_stop
    user_name: Union[str, None] = Field(None, description='Internal user name', min_length=1)
    assistant_name: Union[str, None] = Field(None, description='Internal assistant name', min_length=1)
    system_name: Union[str, None] = Field(None, description='Internal system name', min_length=1)
    presystem: bool = Field(True, description='Whether to insert default system prompt at the beginning')
    model_config = {'json_schema_extra': {'example': {'messages': [{'role': Role.User.value, 'content': 'hello', 'raw': False}], 'model': 'rwkv', 'stream': False, 'stop': None, 'user_name': None, 'assistant_name': None, 'system_name': None, 'presystem': True, 'max_tokens': 1000, 'temperature': 1, 'top_p': 0.3, 'presence_penalty': 0, 'frequency_penalty': 1}}}