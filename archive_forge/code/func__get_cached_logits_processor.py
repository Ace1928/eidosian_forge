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
@lru_cache(maxsize=32)
def _get_cached_logits_processor(guide: str, tokenizer, mode: GuidedDecodingMode):
    if mode == GuidedDecodingMode.JSON:
        return JSONLogitsProcessor(guide, tokenizer)
    elif mode == GuidedDecodingMode.REGEX or mode == GuidedDecodingMode.CHOICE:
        return RegexLogitsProcessor(guide, tokenizer)
    else:
        raise ValueError(f'Unknown guided decoding mode {mode}')