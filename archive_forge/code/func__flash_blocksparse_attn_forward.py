import flash_attn_cuda
import torch
import torch.nn as nn
def _flash_blocksparse_attn_forward(qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal, return_softmax):
    context, softmax_lse, *rest = flash_attn_cuda.fwd_block(qkv, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal, return_softmax, None)
    S_dmask = rest[0] if return_softmax else None
    return (context, softmax_lse, S_dmask)