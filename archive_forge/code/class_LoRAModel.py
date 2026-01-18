import copy
import json
import logging
import math
import os
import re
from typing import (Any, Callable, Dict, Hashable, List, Optional, Tuple, Type)
import safetensors.torch
import torch
from torch import nn
from vllm.config import LoRAConfig
from vllm.utils import LRUCache, in_wsl
from vllm.lora.layers import BaseLayerWithLoRA, LoRAMapping, from_layer, from_layer_sampler
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.utils import parse_fine_tuned_lora_name, replace_submodule
class LoRAModel:
    """A LoRA fine-tuned model."""

    def __init__(self, lora_model_id: int, rank: int, loras: Dict[str, LoRALayerWeights]) -> None:
        self.id = lora_model_id
        assert lora_model_id > 0, f'a valid lora id should be greater than 0, got {self.id}'
        self.rank = rank
        self.loras: Dict[str, LoRALayerWeights] = loras

    @property
    def extra_vocab_size(self) -> int:
        return max((lora.extra_vocab_size for lora in self.loras.values())) if self.loras else 0

    def get_lora(self, module_name: str) -> Optional[LoRALayerWeights]:
        """Get LoRA for a given module by name"""
        return self.loras.get(module_name, None)

    @classmethod
    def from_lora_tensors(cls, lora_model_id: int, rank: int, lora_alpha: int, tensors: Dict[str, torch.Tensor], device: str='cuda', dtype: Optional[torch.dtype]=None, embeddings: Optional[Dict[str, torch.Tensor]]=None, target_embedding_padding: Optional[int]=None, embedding_modules: Optional[Dict[str, str]]=None, embedding_padding_modules: Optional[List[str]]=None) -> 'LoRAModel':
        """Create a LoRAModel from a dictionary of tensors."""
        pin_memory = str(device) == 'cpu' and (not in_wsl())
        loras: Dict[str, LoRALayerWeights] = {}
        for tensor_name, tensor in tensors.items():
            module_name, is_lora_a = parse_fine_tuned_lora_name(tensor_name)
            if module_name not in loras:
                lora_embeddings_tensor = None
                if embeddings:
                    embeddings_module = next((k for k in embedding_modules if k in module_name), None)
                    if embeddings_module:
                        lora_embeddings_tensor = embeddings[embedding_modules[embeddings_module]].to(device=device, dtype=dtype)
                        if pin_memory:
                            lora_embeddings_tensor = lora_embeddings_tensor.pin_memory()
                loras[module_name] = LoRALayerWeights(module_name, rank, lora_alpha, None, None, lora_embeddings_tensor)
            if is_lora_a:
                loras[module_name].lora_a = tensor.to(device=device, dtype=dtype).t()
                if pin_memory:
                    loras[module_name].lora_a = loras[module_name].lora_a.pin_memory()
            else:
                loras[module_name].lora_b = tensor.to(device=device, dtype=dtype).t()
                if any((name in module_name for name in embedding_padding_modules)) and target_embedding_padding is not None:
                    lora_b = loras[module_name].lora_b
                    assert target_embedding_padding >= lora_b.shape[1]
                    addition = target_embedding_padding - lora_b.shape[1]
                    loras[module_name].lora_b = torch.nn.functional.pad(lora_b, (0, addition))
                if pin_memory:
                    loras[module_name].lora_b = loras[module_name].lora_b.pin_memory()
        for lora in loras.values():
            lora.optimize()
        return cls(lora_model_id, rank, loras)

    @classmethod
    def from_local_checkpoint(cls, lora_dir: str, lora_model_id: Optional[int]=None, device: str='cuda', dtype: Optional[torch.dtype]=None, target_embedding_padding: Optional[int]=None, embedding_modules: Optional[Dict[str, str]]=None, embedding_padding_modules: Optional[List[str]]=None) -> 'LoRAModel':
        """Create a LoRAModel from a local checkpoint."""
        lora_config_path = os.path.join(lora_dir, 'adapter_config.json')
        lora_tensor_path = os.path.join(lora_dir, 'adapter_model.safetensors')
        lora_bin_file_path = os.path.join(lora_dir, 'adapter_model.bin')
        new_embeddings_tensor_path = os.path.join(lora_dir, 'new_embeddings.safetensors')
        new_embeddings_bin_file_path = os.path.join(lora_dir, 'new_embeddings.bin')
        if os.path.isfile(lora_tensor_path):
            tensors = safetensors.torch.load_file(lora_tensor_path)
        elif os.path.isfile(lora_bin_file_path):
            tensors = torch.load(lora_bin_file_path)
        else:
            raise ValueError(f"{lora_dir} doesn't contain tensors")
        embeddings = None
        if os.path.isfile(new_embeddings_tensor_path):
            embeddings = safetensors.torch.load_file(new_embeddings_tensor_path)
        elif os.path.isfile(new_embeddings_bin_file_path):
            embeddings = torch.load(new_embeddings_bin_file_path)
        with open(lora_config_path) as f:
            config = json.load(f)
        rank = config['r']
        lora_alpha = config['lora_alpha']
        return cls.from_lora_tensors(lora_model_id=get_lora_id() if lora_model_id is None else lora_model_id, rank=rank, lora_alpha=lora_alpha, tensors=tensors, device=device, dtype=dtype, embeddings=embeddings, target_embedding_padding=target_embedding_padding, embedding_modules=embedding_modules, embedding_padding_modules=embedding_padding_modules)