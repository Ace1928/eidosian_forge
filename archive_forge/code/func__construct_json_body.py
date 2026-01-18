import logging
from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def _construct_json_body(self, prompt: str, params: dict) -> dict:
    return {'prompt': prompt, **params}