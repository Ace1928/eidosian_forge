import math
import hydra
import torch
import torch.nn as nn
from einops import rearrange
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from flash_attn.flash_blocksparse_attn_interface import (
class FlashBlocksparseAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """

    def __init__(self, sparsity_config, softmax_temp=None, attention_dropout=0.0, max_seq_length=2048, device=None, dtype=None):
        super().__init__()
        self.sparsity_config = hydra.utils.instantiate(sparsity_config)
        self.softmax_temp = softmax_temp
        self.dropout_p = attention_dropout
        max_seq_length = (max_seq_length + 256 - 1) // 256 * 256
        layout = self.sparsity_config.make_layout(max_seq_length)
        self.register_buffer('layout', layout)
        blockmask_converted = convert_blockmask(self.layout, causal=False)
        self.register_buffer('blockmask_converted', blockmask_converted)

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, causal=False, cu_seqlens=None, max_s=None, need_weights=False, convert_mask=True):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            key_padding_mask: An implementation of BaseMask that encodes how
                         many query each sequence in the batch consists of
        """
        assert not need_weights
        assert attn_mask is None
        assert qkv.dtype == torch.float16
        assert qkv.is_cuda
        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            seqlen_rounded = (seqlen + 256 - 1) // 256 * 256
            assert seqlen_rounded // 16 <= self.layout.shape[0], seqlen_rounded // 256 <= self.layout.shape[1]
            blockmask = self.layout[:seqlen_rounded // 16, :seqlen_rounded // 256]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=qkv.device)
                output = flash_blocksparse_attn_func(qkv, cu_seqlens, blockmask, self.dropout_p if self.training else 0.0, max_s, softmax_scale=self.softmax_temp, causal=causal)
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                key_padding_mask_bool = key_padding_mask.bool_matrix
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask_bool)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_blocksparse_attn_func(x_unpad, cu_seqlens, blockmask, self.dropout_p if self.training else 0.0, max_s, softmax_scale=self.softmax_temp, causal=causal)
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices, batch_size, seqlen), 'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            seqlen = max_s
            seqlen_rounded = (seqlen + 256 - 1) // 256 * 256
            assert seqlen_rounded // 16 <= self.layout.shape[0], seqlen_rounded // 256 <= self.layout.shape[1]
            blockmask = self.layout[:seqlen_rounded // 16, :seqlen_rounded // 256]
            if convert_mask:
                output = flash_blocksparse_attn_func(qkv, cu_seqlens, blockmask, self.dropout_p if self.training else 0.0, max_s, softmax_scale=self.softmax_temp, causal=causal)
            else:
                output = flash_blocksparse_attn_func(qkv, cu_seqlens, self.blockmask_converted, self.dropout_p if self.training else 0.0, max_s, softmax_scale=self.softmax_temp, causal=causal, convert_mask=False)
        return (output, None)