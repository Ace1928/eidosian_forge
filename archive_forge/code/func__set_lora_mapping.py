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
def _set_lora_mapping(self, mapping: LoRAMapping) -> None:
    base_indices, sampler_indices, sampler_indices_padded, embeddings_indices, indices_len = convert_mapping(mapping, self.lora_index_to_id, self.lora_slots + 1, self.vocab_size, self.lora_config.lora_extra_vocab_size)
    self.base_indices[:base_indices.shape[0]].copy_(base_indices)
    self.sampler_indices[:sampler_indices.shape[0]].copy_(sampler_indices)
    self.sampler_indices_padded[:sampler_indices_padded.shape[0]].copy_(sampler_indices_padded)
    self.embeddings_indices[:embeddings_indices.shape[0], :embeddings_indices.shape[1]].copy_(embeddings_indices)
    self.indices_len[:] = indices_len