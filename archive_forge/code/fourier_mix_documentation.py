import torch
from torch.cuda.amp import autocast
from xformers.components.attention import Attention, AttentionConfig, register_attention

        FFT-based pseudo-attention mechanism, from
        "
        "FNet: Mixing Tokens with Fourier Transforms"
        Lee-Thorp et al., 2021, https://arxiv.org/pdf/2105.03824.pdf
        