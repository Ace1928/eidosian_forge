import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ..auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .configuration_instructblip import InstructBlipConfig, InstructBlipQFormerConfig, InstructBlipVisionConfig
def _preprocess_accelerate(self):
    """
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
    hf_device_map = self.hf_device_map
    if len(hf_device_map) > 1 and 'language_model' not in hf_device_map and (torch.cuda.device_count() > 1):
        logger.warning('The `language_model` is not in the `hf_device_map` dictionary and you are running your script in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`. Please pass a `device_map` that contains `language_model` to remove this warning. Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for more details on creating a `device_map` for large models.')
    if hasattr(self.language_model, '_hf_hook'):
        self.language_model._hf_hook.io_same_device = True