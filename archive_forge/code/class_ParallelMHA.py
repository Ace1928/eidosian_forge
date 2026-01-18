import math
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn.utils.distributed import get_dim_for_local_rank
class ParallelMHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(self, embed_dim, num_heads, process_group, num_heads_kv=None, qkv_proj_bias=True, out_proj_bias=True, dropout=0.0, softmax_scale=None, causal=False, layer_idx=None, rotary_emb_dim=0, rotary_emb_base=10000.0, rotary_emb_scale_base=None, rotary_emb_interleaved=False, use_alibi=False, window_size=(-1, -1), use_flash_attn=False, checkpointing=False, sequence_parallel=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.checkpointing = checkpointing
        self.process_group = process_group
        self.world_size = process_group.size()
        self.local_rank = torch.distributed.get_rank(process_group)
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0, 'embed_dim must be divisible by num_heads'
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert self.num_heads % self.num_heads_kv == 0, 'num_heads must be divisible by num_heads_kv'
        self.num_heads_per_rank = get_dim_for_local_rank(self.num_heads, self.world_size, self.local_rank)
        self.num_heads_kv_per_rank = get_dim_for_local_rank(self.num_heads_kv, self.world_size, self.local_rank)
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        if use_alibi:
            assert use_flash_attn, 'ALiBi code path requires flash_attn'
            num_heads_local = math.ceil(self.num_heads / self.world_size)
            alibi_slopes = torch.tensor(get_alibi_slopes(num_heads)[self.local_rank * num_heads_local:(self.local_rank + 1) * num_heads_local], device=device)
        else:
            alibi_slopes = None
        if window_size != (-1, -1):
            assert use_flash_attn, 'Local (sliding window) attention code path requires flash_attn'
        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, 'rotary_emb is not installed'
            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, base=rotary_emb_base, scale_base=rotary_emb_scale_base, interleaved=rotary_emb_interleaved, device=device)
        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError('fused_dense is not installed')
        self.Wqkv = ColumnParallelLinear(embed_dim, qkv_dim, process_group, bias=qkv_proj_bias, sequence_parallel=sequence_parallel, multiple_of=self.head_dim * (self.num_heads // self.num_heads_kv + 2), **factory_kwargs)
        inner_attn_cls = partial(FlashSelfAttention, alibi_slopes=alibi_slopes, window_size=window_size) if use_flash_attn else SelfAttention
        inner_cross_attn_cls = partial(FlashCrossAttention, alibi_slopes=alibi_slopes, window_size=window_size) if use_flash_attn else CrossAttention
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = inner_cross_attn_cls(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.out_proj = RowParallelLinear(embed_dim, embed_dim, process_group, bias=out_proj_bias, sequence_parallel=sequence_parallel, multiple_of=self.head_dim, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(batch_size, max_seqlen, 2, self.num_heads_kv_per_rank, self.head_dim, dtype=dtype, device=device)

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert self.layer_idx is not None, 'Generation requires layer_idx in the constructor'
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        assert inference_params is not None and inference_params.seqlen_offset > 0
        assert self.use_flash_attn
        if self.rotary_emb_dim > 0:
            assert self.rotary_emb.scale is None, 'This code path does not support xPos'
            self.rotary_emb._update_cos_sin_cache(inference_params.max_seqlen, device=q.device, dtype=q.dtype)
            rotary_cos, rotary_sin = (self.rotary_emb._cos_cached, self.rotary_emb._sin_cached)
        else:
            rotary_cos, rotary_sin = (None, None)
        batch = q.shape[0]
        kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        cache_seqlens = inference_params.lengths_per_sample[:batch] if inference_params.lengths_per_sample is not None else inference_params.seqlen_offset
        alibi_slopes = getattr(self.inner_cross_attn, 'alibi_slopes', None)
        context = flash_attn_with_kvcache(q, kv_cache[:, :, 0], kv_cache[:, :, 1], kv[:, :, 0], kv[:, :, 1], rotary_cos=rotary_cos, rotary_sin=rotary_sin, cache_seqlens=cache_seqlens, softmax_scale=self.inner_cross_attn.softmax_scale, causal=self.inner_cross_attn.causal, rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False, alibi_slopes=alibi_slopes)
        return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if inference_params.seqlen_offset == 0 or not self.use_flash_attn:
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            batch = q.shape[0]
            kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            cache_seqlens = inference_params.lengths_per_sample[:batch] if inference_params.lengths_per_sample is not None else inference_params.seqlen_offset
            alibi_slopes = getattr(self.inner_cross_attn, 'alibi_slopes', None)
            context = flash_attn_with_kvcache(q, kv_cache[:, :, 0], kv_cache[:, :, 1], kv[:, :, 0], kv[:, :, 1], cache_seqlens=cache_seqlens, softmax_scale=self.inner_cross_attn.softmax_scale, causal=self.inner_cross_attn.causal, alibi_slopes=alibi_slopes)
            return context

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.Wqkv(x)
        if seqlen is not None:
            qkv = rearrange(qkv, '(b s) ... -> b s ...', s=seqlen)
        seqlen_offset = 0 if inference_params is None else inference_params.lengths_per_sample if inference_params.lengths_per_sample is not None else inference_params.seqlen_offset
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        if self.num_heads_kv == self.num_heads:
            qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, d=self.head_dim)
            if inference_params is None or inference_params.seqlen_offset == 0 or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0) or (not self.use_flash_attn):
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen)
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_attn(qkv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
                else:
                    context = self._update_kvcache_attention(qkv[:, :, 0], qkv[:, :, 1:], inference_params)
            else:
                context = self._apply_rotary_update_kvcache_attention(qkv[:, :, 0], qkv[:, :, 1:], inference_params)
        else:
            q = rearrange(qkv[..., :self.num_heads_per_rank * self.head_dim], '... (h d) -> ... h d', d=self.head_dim)
            kv = rearrange(qkv[..., self.num_heads_per_rank * self.head_dim:], '... (two hkv d) -> ... two hkv d', two=2, d=self.head_dim)
            if inference_params is None or inference_params.seqlen_offset == 0 or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0) or (not self.use_flash_attn):
                if self.rotary_emb_dim > 0:
                    q, kv = self.rotary_emb(q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen)
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_cross_attn(q, kv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_cross_attn, q, kv, **kwargs)
                else:
                    context = self._update_kvcache_attention(q, kv, inference_params)
            else:
                context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
        context = rearrange(context, 'b s h d -> b s (h d)')
        if seqlen is not None:
            context = rearrange(context, 'b s d -> (b s) d')
        out = self.out_proj(context)
        return out