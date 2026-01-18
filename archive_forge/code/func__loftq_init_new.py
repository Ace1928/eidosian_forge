from __future__ import annotations
import logging
import os
from typing import Callable, Optional, Union
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from safetensors import SafetensorError, safe_open
from transformers.utils import cached_file
from transformers.utils.hub import get_checkpoint_shard_files
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
@torch.no_grad()
def _loftq_init_new(qweight, weight, num_bits: int, reduced_rank: int):
    if num_bits != 4:
        raise ValueError('Only 4 bit quantization supported at the moment.')
    if not is_bnb_4bit_available():
        raise ValueError('bitsandbytes 4bit quantization is not available.')
    compute_device = 'cuda'
    dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, qweight.quant_state)
    weight = weight.to(device=compute_device, dtype=torch.float32)
    residual = weight - dequantized_weight
    torch.cuda.empty_cache()
    output = _low_rank_decomposition(residual, reduced_rank=reduced_rank)
    L, R, reduced_rank = (output['L'], output['R'], output['reduced_rank'])
    return (R, L)