import asyncio
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional, Union
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (CompletionRequest,
from vllm.lora.request import LoRARequest
def _validate_prompt_and_tokenize(self, request: Union[ChatCompletionRequest, CompletionRequest], prompt: Optional[str]=None, prompt_ids: Optional[List[int]]=None) -> List[int]:
    if not (prompt or prompt_ids):
        raise ValueError('Either prompt or prompt_ids should be provided.')
    if prompt and prompt_ids:
        raise ValueError('Only one of prompt or prompt_ids should be provided.')
    input_ids = prompt_ids if prompt_ids is not None else self.tokenizer(prompt).input_ids
    token_num = len(input_ids)
    if request.max_tokens is None:
        request.max_tokens = self.max_model_len - token_num
    if token_num + request.max_tokens > self.max_model_len:
        raise ValueError(f"This model's maximum context length is {self.max_model_len} tokens. However, you requested {request.max_tokens + token_num} tokens ({token_num} in the messages, {request.max_tokens} in the completion). Please reduce the length of the messages or completion.")
    else:
        return input_ids