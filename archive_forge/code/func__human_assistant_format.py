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
def _human_assistant_format(input_text: str) -> str:
    if input_text.count('Human:') == 0 or (input_text.find('Human:') > input_text.find('Assistant:') and 'Assistant:' in input_text):
        input_text = HUMAN_PROMPT + ' ' + input_text
    if input_text.count('Assistant:') == 0:
        input_text = input_text + ASSISTANT_PROMPT
    if input_text[:len('Human:')] == 'Human:':
        input_text = '\n\n' + input_text
    input_text = _add_newlines_before_ha(input_text)
    count = 0
    for i in range(len(input_text)):
        if input_text[i:i + len(HUMAN_PROMPT)] == HUMAN_PROMPT:
            if count % 2 == 0:
                count += 1
            else:
                warnings.warn(ALTERNATION_ERROR + f' Received {input_text}')
        if input_text[i:i + len(ASSISTANT_PROMPT)] == ASSISTANT_PROMPT:
            if count % 2 == 1:
                count += 1
            else:
                warnings.warn(ALTERNATION_ERROR + f' Received {input_text}')
    if count % 2 == 1:
        input_text = input_text + ASSISTANT_PROMPT
    return input_text