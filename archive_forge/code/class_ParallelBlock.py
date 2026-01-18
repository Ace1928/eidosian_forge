from functools import partial
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import StochasticDepth
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp
class ParallelBlock(nn.Module):
    """The attention (mixer) and MLP blocks are done in parallel, similar to GPT-J, GPT-NeoX,
    and PaLM.
    """

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm, dropout_cls=nn.Dropout, resid_dropout1=0.0, resid_dropout2=0.0, tied_norm=False, fused_dropout_add_ln=False, residual_in_fp32=False, sequence_parallel=False, mark_shared_params=False):
        """
        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA / MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA / MLP, returning both
        the hidden_states (output1 of the MHA / MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.tied_norm = tied_norm
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.residual_in_fp32 = residual_in_fp32
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        self.dropout2 = dropout_cls(resid_dropout2)
        if not self.tied_norm:
            self.norm2 = norm_cls(dim)
        if self.fused_dropout_add_ln:
            assert layer_norm_fn is not None, 'Triton is not installed'
            assert isinstance(self.norm1, (nn.LayerNorm, RMSNorm)) and isinstance(self.dropout1, nn.Dropout)
        if sequence_parallel:
            for p in self.norm1.parameters():
                p._sequence_parallel = True
            if hasattr(self, 'norm2'):
                for p in self.norm2.parameters():
                    p._sequence_parallel = True
        if mark_shared_params:
            for p in self.norm1.parameters():
                p._shared_params = True
            if hasattr(self, 'norm2'):
                for p in self.norm2.parameters():
                    p._shared_params = True

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, hidden_states1: Tensor, hidden_states2: Optional[Tensor]=None, residual: Optional[Tensor]=None, mixer_kwargs=None):
        """Pass the input through the encoder layer.

        Args:
            hidden_states1: the output of the previous attention (mixer) or embedding layer.
            hidden_states2: the output of the previous MLP layer (if None, will use hidden_states1).
            residual.
        """
        if not self.fused_dropout_add_ln:
            dropped1 = self.dropout1(hidden_states1)
            if hidden_states2 is not None:
                dropped2 = self.dropout2(hidden_states2)
                residual = residual + dropped1 + dropped2 if residual is not None else dropped1 + dropped2
            else:
                residual = residual + dropped1 if residual is not None else dropped1
            hidden_states1 = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            hidden_states2 = self.norm2(residual.to(dtype=self.norm2.weight.dtype)) if not self.tied_norm else hidden_states1
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            weight2, bias2 = (self.norm2.weight, self.norm2.bias) if not self.tied_norm else (None, None)
            hidden_states1, *rest, residual = layer_norm_fn(hidden_states1, self.norm1.weight, self.norm1.bias, residual=residual, x1=hidden_states2, weight1=weight2, bias1=bias2, eps=self.norm1.eps, dropout_p=self.dropout1.p if self.training else 0.0, prenorm=True, residual_in_fp32=self.residual_in_fp32, is_rms_norm=isinstance(self.norm1, RMSNorm))
            if self.tied_norm:
                hidden_states2 = hidden_states1
            else:
                hidden_states2, = rest
        if mixer_kwargs is None:
            mixer_kwargs = {}
        hidden_states1 = self.mixer(hidden_states1, **mixer_kwargs)
        hidden_states2 = self.mlp(hidden_states2)
        return (hidden_states1, hidden_states2, residual)