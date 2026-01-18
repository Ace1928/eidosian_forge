from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.utils.data import Dataset
from .generation.configuration_utils import GenerationConfig
from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .trainer import Trainer
from .utils import logging
def _pad_tensors_to_max_len(self, tensor, max_length):
    if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id'):
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
    elif self.model.config.pad_token_id is not None:
        pad_token_id = self.model.config.pad_token_id
    else:
        raise ValueError('Pad_token_id must be set in the configuration of the model, in order to pad tensors')
    padded_tensor = pad_token_id * torch.ones((tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:, :tensor.shape[-1]] = tensor
    return padded_tensor