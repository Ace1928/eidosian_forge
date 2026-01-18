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
def activate_lora(self, lora_id: int) -> bool:
    if lora_id not in self._active_loras and len(self._active_loras) >= self.lora_slots:
        self._active_loras.remove_oldest()
    result = super().activate_lora(lora_id)
    self._active_loras.touch(lora_id)
    return result