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
def _create_lora_modules(self):
    for module_name, module in self.model.named_modules():
        if not self._match_target_modules(module_name):
            continue
        new_module = replace_submodule(self.model, module_name, from_layer(module, self.lora_slots, self.lora_config, self.model.config))
        if 'lm_head' in module_name:
            sampler_module = self.model.get_submodule('sampler')
            new_module = replace_submodule(self.model, 'sampler', from_layer_sampler(sampler_module, module, self.lora_slots, self.lora_config, self.model.config))
        self.register_module(module_name, new_module)
        self._register_packed_modules(module_name)
        new_module.set_mapping(self.base_indices, self.sampler_indices, self.sampler_indices_padded, self.embeddings_indices, self.indices_len)