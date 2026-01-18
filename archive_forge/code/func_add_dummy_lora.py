import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, List, Optional, Set, Type
import torch
from vllm.lora.models import (LoRAModel, LoRAModelManager,
from vllm.lora.request import LoRARequest
from vllm.lora.layers import LoRAMapping
from vllm.config import LoRAConfig
def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
    if lora_request.lora_int_id in self.list_loras():
        return False
    return self._lora_manager.add_lora(self._lora_manager.create_dummy_lora(lora_request.lora_int_id, rank, self.embedding_modules))