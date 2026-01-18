import math
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn.utils.distributed import get_dim_for_local_rank
class FlashCrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0, alibi_slopes=None, window_size=(-1, -1), deterministic=False):
        super().__init__()
        assert flash_attn_varlen_kvpacked_func is not None, 'FlashAttention is not installed'
        assert flash_attn_kvpacked_func is not None, 'FlashAttention is not installed'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.register_buffer('alibi_slopes', alibi_slopes, persistent=False)
        self.window_size = window_size
        self.deterministic = deterministic

    def forward(self, q, kv, causal=None, cu_seqlens=None, max_seqlen=None, cu_seqlens_k=None, max_seqlen_k=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
            max_seqlen: int. Maximum sequence length in the batch of q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into kv.
            max_seqlen_k: int. Maximum sequence length in the batch of k and v.
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            assert cu_seqlens_k is not None
            assert cu_seqlens_k.dtype == torch.int32
            assert max_seqlen_k is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens, cu_seqlens_k, max_seqlen, max_seqlen_k, self.drop.p if self.training else 0.0, softmax_scale=self.softmax_scale, causal=causal, alibi_slopes=self.alibi_slopes, window_size=self.window_size, deterministic=self.deterministic)
        else:
            batch_size, seqlen_q = (q.shape[0], q.shape[1])
            seqlen_k = kv.shape[1]
            assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
            return flash_attn_kvpacked_func(q, kv, self.drop.p if self.training else 0.0, causal=causal, softmax_scale=self.softmax_scale, alibi_slopes=self.alibi_slopes, window_size=self.window_size, deterministic=self.deterministic)