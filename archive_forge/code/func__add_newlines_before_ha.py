import asyncio
import json
import warnings
from abc import ABC
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
from langchain_community.utilities.anthropic import (
def _add_newlines_before_ha(input_text: str) -> str:
    new_text = input_text
    for word in ['Human:', 'Assistant:']:
        new_text = new_text.replace(word, '\n\n' + word)
        for i in range(2):
            new_text = new_text.replace('\n\n\n' + word, '\n\n' + word)
    return new_text