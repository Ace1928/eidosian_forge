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
class EmbeddingsBody(BaseModel):
    input: Union[str, List[str], List[List[int]], None]
    model: Union[str, None] = 'rwkv'
    encoding_format: str = None
    fast_mode: bool = False
    model_config = {'json_schema_extra': {'example': {'input': 'a big apple', 'model': 'rwkv', 'encoding_format': None, 'fast_mode': False}}}