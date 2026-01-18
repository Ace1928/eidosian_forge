import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, List, Optional, Set, Type
import torch
from vllm.lora.models import (LoRAModel, LoRAModelManager,
from vllm.lora.request import LoRARequest
from vllm.lora.layers import LoRAMapping
from vllm.config import LoRAConfig
class LRUCacheWorkerLoRAManager(WorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Uses an LRU Cache. Every request, the requested LoRAs will be loaded
    (unless they are already loaded) and least recently used LoRAs will
    be unloaded if the cache is above capacity."""
    _lora_manager_cls: Type[LRUCacheLoRAModelManager] = LRUCacheLoRAModelManager

    def create_lora_manager(self, model: torch.nn.Module) -> Any:
        lora_manager = create_lora_manager(model, lora_manager_cls=self._lora_manager_cls, max_num_seqs=self.max_num_seqs, vocab_size=self.vocab_size, lora_config=self.lora_config, max_num_batched_tokens=self.max_num_batched_tokens)
        self._lora_manager: LRUCacheLoRAModelManager = lora_manager
        return lora_manager.model

    def _apply_loras(self, lora_requests: List[LoRARequest]) -> None:
        loras_map = {lora_request.lora_int_id: lora_request for lora_request in lora_requests if lora_request}
        if len(loras_map) > self._lora_manager.lora_slots:
            raise RuntimeError(f'Number of requested LoRAs ({len(loras_map)}) is greater than the number of GPU LoRA slots ({self._lora_manager.lora_slots}).')
        for lora in loras_map.values():
            self.add_lora(lora)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if lora_request.lora_int_id not in self.list_loras():
            if len(self._lora_manager) + 1 > self._lora_manager.capacity:
                self._lora_manager.remove_oldest_lora()
            lora = self._load_lora(lora_request)
            loaded = self._lora_manager.add_lora(lora)
        else:
            loaded = self._lora_manager.get_lora(lora_request.lora_int_id)
        self._lora_manager.activate_lora(lora_request.lora_int_id)
        return loaded