import flash_attn_cuda
import torch
import torch.nn as nn
def _flash_blocksparse_attn_backward(dout, qkv, out, S_dmask, softmax_lse, cu_seqlens, blockmask, dropout_p, max_s, softmax_scale, causal):
    dqkv, dp, softmax_d = flash_attn_cuda.bwd_block(dout, qkv, out, S_dmask, softmax_lse, cu_seqlens, blockmask, dropout_p, softmax_scale, max_s, causal, None)
    return dqkv