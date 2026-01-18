import asyncio
import concurrent.futures
from copy import copy
from enum import Enum
from functools import lru_cache
from json import dumps as json_dumps
from re import escape as regex_escape
from typing import Union, Tuple
from pydantic import BaseModel
from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest
from vllm.model_executor.guided_logits_processors import JSONLogitsProcessor, RegexLogitsProcessor
def _get_guide_and_mode(request: Union[CompletionRequest, ChatCompletionRequest]) -> Tuple[str, GuidedDecodingMode]:
    if request.guided_json:
        if not isinstance(request.guided_json, (str, dict, BaseModel)):
            raise TypeError('JSON schema must be str, dict, or BaseModel')
        json = request.guided_json
        if isinstance(json, dict):
            json = json_dumps(json, sort_keys=True)
        elif isinstance(json, BaseModel):
            json = str(json.__signature__)
        return (json, GuidedDecodingMode.JSON)
    elif request.guided_regex:
        if not isinstance(request.guided_regex, str):
            raise TypeError('Regex must be string')
        return (request.guided_regex, GuidedDecodingMode.REGEX)
    elif request.guided_choice:
        if not isinstance(request.guided_choice, list):
            raise TypeError('Choices must be a list')
        choices = [regex_escape(str(choice)) for choice in request.guided_choice]
        choices_regex = '(' + '|'.join(choices) + ')'
        return (choices_regex, GuidedDecodingMode.CHOICE)
    else:
        return (None, None)