import torch
from torch.cuda.amp import autocast
from xformers.components.attention import Attention, AttentionConfig, register_attention
@register_attention('fourier_mix', AttentionConfig)
class FourierMix(Attention):

    def __init__(self, dropout: float, *_, **__):
        """
        FFT-based pseudo-attention mechanism, from
        "
        "FNet: Mixing Tokens with Fourier Transforms"
        Lee-Thorp et al., 2021, https://arxiv.org/pdf/2105.03824.pdf
        """
        super().__init__()
        self.attn_drop = torch.nn.Dropout(dropout, inplace=False)
        self.supports_attention_mask = False
        self.requires_input_projection = False

    def forward(self, q: torch.Tensor, *_, **__):
        with autocast(enabled=False):
            att = torch.fft.fft2(q).real
        att = self.attn_drop(att)
        return att