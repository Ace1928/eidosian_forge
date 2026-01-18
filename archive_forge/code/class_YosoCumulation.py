import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_yoso import YosoConfig
class YosoCumulation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, query_mask, key_mask, query, key, value, config):
        hash_code_len = config['hash_code_len']
        expectation = (1 - torch.acos(torch.matmul(query, key.transpose(-1, -2))) / math.pi) ** hash_code_len
        expectation = expectation * query_mask[:, :, None] * key_mask[:, None, :]
        cumulation_value = torch.matmul(expectation, value)
        ctx.save_for_backward(query_mask, key_mask, expectation, query, key, value)
        ctx.config = config
        return cumulation_value

    @staticmethod
    def backward(ctx, grad):
        grad = to_contiguous(grad)
        query_mask, key_mask, expectation, query, key, value = ctx.saved_tensors
        config = ctx.config
        hash_code_len = config['hash_code_len']
        weighted_exp = torch.matmul(grad, value.transpose(-1, -2)) * expectation
        grad_query = torch.matmul(weighted_exp, hash_code_len / 2 * key)
        grad_key = torch.matmul(weighted_exp.transpose(-1, -2), hash_code_len / 2 * query)
        grad_value = torch.matmul(expectation.transpose(-1, -2), grad)
        return (None, None, grad_query, grad_key, grad_value, None)