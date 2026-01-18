import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, List, Optional, Set, Type
import torch
from vllm.lora.models import (LoRAModel, LoRAModelManager,
from vllm.lora.request import LoRARequest
from vllm.lora.layers import LoRAMapping
from vllm.config import LoRAConfig
def _load_lora(self, lora_request: LoRARequest) -> LoRAModel:
    try:
        lora = self._lora_model_cls.from_local_checkpoint(lora_request.lora_local_path, lora_model_id=lora_request.lora_int_id, device='cpu', dtype=self.lora_config.lora_dtype, target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size, embedding_modules=self.embedding_modules, embedding_padding_modules=self.embedding_padding_modules)
    except Exception as e:
        raise RuntimeError(f'Loading lora {lora_request.lora_local_path} failed') from e
    if lora.rank > self.lora_config.max_lora_rank:
        raise ValueError(f'LoRA rank {lora.rank} is greater than max_lora_rank {self.lora_config.max_lora_rank}.')
    if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
        raise ValueError(f'LoRA added vocab size {lora.extra_vocab_size} is greater than lora_extra_vocab_size {self.lora_config.lora_extra_vocab_size}.')
    return lora