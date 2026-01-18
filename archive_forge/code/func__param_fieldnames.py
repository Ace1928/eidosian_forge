from __future__ import annotations
import json
from io import StringIO
from typing import Any, Dict, Iterator, List, Optional
import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra
from langchain_core.utils import get_pydantic_field_names
@property
def _param_fieldnames(self) -> List[str]:
    ignore_keys = ['base_url', 'cache', 'callback_manager', 'callbacks', 'metadata', 'name', 'request_timeout', 'streaming', 'tags', 'verbose']
    attrs = [k for k in get_pydantic_field_names(self.__class__) if k not in ignore_keys]
    return attrs