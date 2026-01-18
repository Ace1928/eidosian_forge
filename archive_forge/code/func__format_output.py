import logging
from typing import Any, Dict, List, Literal, Optional
from aiohttp import ClientSession
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
from langchain_community.utilities.requests import Requests
def _format_output(self, output: dict) -> str:
    if self.feature == 'text':
        return output[self.provider]['generated_text']
    else:
        return output[self.provider]['items'][0]['image']