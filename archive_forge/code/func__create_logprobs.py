import asyncio
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional, Union
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (CompletionRequest,
from vllm.lora.request import LoRARequest
def _create_logprobs(self, token_ids: List[int], top_logprobs: Optional[List[Optional[Dict[int, float]]]]=None, num_output_top_logprobs: Optional[int]=None, initial_text_offset: int=0) -> LogProbs:
    """Create OpenAI-style logprobs."""
    logprobs = LogProbs()
    last_token_len = 0
    if num_output_top_logprobs:
        logprobs.top_logprobs = []
    for i, token_id in enumerate(token_ids):
        step_top_logprobs = top_logprobs[i]
        if step_top_logprobs is not None:
            token_logprob = step_top_logprobs[token_id]
        else:
            token_logprob = None
        token = self.tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(token_logprob)
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
        last_token_len = len(token)
        if num_output_top_logprobs:
            logprobs.top_logprobs.append({self.tokenizer.convert_ids_to_tokens(i): p for i, p in step_top_logprobs.items()} if step_top_logprobs else None)
    return logprobs