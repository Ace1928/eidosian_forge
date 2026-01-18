import asyncio
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional, Union
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (CompletionRequest,
from vllm.lora.request import LoRARequest
def _maybe_get_lora(self, request) -> Optional[LoRARequest]:
    if request.model == self.served_model:
        return
    for lora in self.lora_requests:
        if request.model == lora.lora_name:
            return lora
    raise ValueError('The model `{request.model}` does not exist.')