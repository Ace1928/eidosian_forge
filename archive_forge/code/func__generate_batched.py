import inspect
import math
import os
import time
import typing
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Union
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object, is_deepspeed_available
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import (
from ..core import (
from ..import_utils import is_npu_available, is_torch_greater_2_0, is_xpu_available
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper, create_reference_model
from . import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig, RunningMoments
from transformers import pipeline
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
def _generate_batched(self, model: PreTrainedModelWrapper, query_tensors: List[torch.Tensor], length_sampler: Optional[Callable]=None, batch_size: int=4, return_prompt: bool=True, pad_to_multiple_of: Optional[int]=None, remove_padding: bool=True, **generation_kwargs):
    outputs = []
    padding_side_default = self.tokenizer.padding_side
    if not self.is_encoder_decoder:
        self.tokenizer.padding_side = 'left'
    batch_size = min(len(query_tensors), batch_size)
    for i in range(0, len(query_tensors), batch_size):
        if length_sampler is not None:
            generation_kwargs['max_new_tokens'] = length_sampler()
        end_index = min(len(query_tensors), i + batch_size)
        batch = query_tensors[i:end_index]
        batch_mask = [torch.ones_like(element) for element in batch]
        inputs = {'input_ids': batch, 'attention_mask': batch_mask}
        padded_inputs = self.tokenizer.pad(inputs, padding=True, max_length=None, pad_to_multiple_of=pad_to_multiple_of, return_tensors='pt').to(self.current_device)
        generations = self.accelerator.unwrap_model(model).generate(**padded_inputs, **generation_kwargs)
        for generation, mask in zip(generations, padded_inputs['attention_mask']):
            if not self.is_encoder_decoder:
                output = generation[(1 - mask).sum():]
            else:
                output = generation
            if not return_prompt and (not self.is_encoder_decoder):
                output = output[mask.sum():]
            if remove_padding and self.tokenizer.eos_token_id in output:
                pad_mask = output == self.tokenizer.eos_token_id
                pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                output = output[:pad_start + 1]
            outputs.append(output)
    self.tokenizer.padding_side = padding_side_default
    return outputs