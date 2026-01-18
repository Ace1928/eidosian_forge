import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, List, Optional, Set, Type
import torch
from vllm.lora.models import (LoRAModel, LoRAModelManager,
from vllm.lora.request import LoRARequest
from vllm.lora.layers import LoRAMapping
from vllm.config import LoRAConfig
def _apply_loras(self, lora_requests: List[LoRARequest]) -> None:
    loras_map = {lora_request.lora_int_id: lora_request for lora_request in lora_requests if lora_request}
    if len(loras_map) > self._lora_manager.lora_slots:
        raise RuntimeError(f'Number of requested LoRAs ({len(loras_map)}) is greater than the number of GPU LoRA slots ({self._lora_manager.lora_slots}).')
    for lora in loras_map.values():
        self.add_lora(lora)