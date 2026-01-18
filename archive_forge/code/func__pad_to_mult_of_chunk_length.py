import sys
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_reformer import ReformerConfig
def _pad_to_mult_of_chunk_length(self, input_ids, inputs_embeds=None, attention_mask=None, position_ids=None, input_shape=None, padding_length=None, padded_seq_length=None, device=None):
    logger.warning_once(f'Input ids are automatically padded from {input_shape[-1]} to {input_shape[-1] + padding_length} to be a multiple of `config.chunk_length`: {padded_seq_length}')
    padded_input_ids = torch.full((input_shape[0], padding_length), self.config.pad_token_id, device=device, dtype=torch.long)
    if attention_mask is not None:
        pad_attention_mask = torch.zeros(input_shape[0], padding_length, device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=-1)
    else:
        attention_mask = torch.cat([torch.ones(input_shape, device=device, dtype=torch.bool), torch.zeros((input_shape[0], padding_length), device=device, dtype=torch.bool)], dim=-1)
    if input_ids is not None:
        input_ids = torch.cat([input_ids, padded_input_ids], dim=-1)
        input_shape = input_ids.size()
        if position_ids is not None:
            padded_position_ids = torch.arange(input_shape[-1], padded_seq_length, dtype=torch.long, device=device)
            padded_position_ids = position_ids.unsqueeze(0).expand(input_shape[0], padding_length)
            position_ids = torch.cat([position_ids, padded_position_ids], dim=-1)
    if inputs_embeds is not None:
        padded_inputs_embeds = self.embeddings(padded_input_ids, position_ids)
        inputs_embeds = torch.cat([inputs_embeds, padded_inputs_embeds], dim=-2)
        input_shape = inputs_embeds.size()
    return (input_ids, inputs_embeds, attention_mask, position_ids, input_shape)