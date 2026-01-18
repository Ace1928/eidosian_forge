import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.cpp_extension import load
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mra import MraConfig
class MraSampledDenseMatMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, dense_query, dense_key, indices, block_size):
        sparse_qk_prod = mm_to_sparse(dense_query, dense_key, indices, block_size)
        ctx.save_for_backward(dense_query, dense_key, indices)
        ctx.block_size = block_size
        return sparse_qk_prod

    @staticmethod
    def backward(ctx, grad):
        dense_query, dense_key, indices = ctx.saved_tensors
        block_size = ctx.block_size
        query_num_block = dense_query.size(1) // block_size
        key_num_block = dense_key.size(1) // block_size
        indices_T = transpose_indices(indices, query_num_block, key_num_block)
        grad_key = sparse_dense_mm(grad.transpose(-1, -2), indices_T, dense_query, key_num_block)
        grad_query = sparse_dense_mm(grad, indices, dense_key, query_num_block)
        return (grad_query, grad_key, None, None)

    @staticmethod
    def operator_call(dense_query, dense_key, indices, block_size=32):
        return MraSampledDenseMatMul.apply(dense_query, dense_key, indices, block_size)