import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
class MegaMultiDimensionDampedEma(nn.Module):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.ndim = config.ema_projection_size
        self.bidirectional = config.bidirectional
        self.truncation = config.truncation
        self.scale = math.sqrt(1.0 / self.ndim)
        kernel_dim = 2 * config.hidden_size if self.bidirectional else config.hidden_size
        self.damping_factor = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        self.decay_factor = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        self.ema_expansion_matrix = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        self.kernel_projection_matrix = nn.Parameter(torch.Tensor(kernel_dim, self.ndim))
        self.residual_weight = nn.Parameter(torch.Tensor(config.hidden_size))
        self._kernel = None
        self._coeffs = None

    def _compute_ema_coefficients(self):
        self._coeffs = None
        damping_factor = torch.sigmoid(self.damping_factor)
        decay_factor = torch.sigmoid(self.decay_factor)
        previous_timestep_weight = 1.0 - damping_factor * decay_factor
        return (damping_factor, previous_timestep_weight)

    def _compute_efficient_ema_kernel(self, length: int):
        self._kernel = None
        damping_factor, previous_timestep_weight = self._compute_ema_coefficients()
        vander = torch.arange(length).to(damping_factor).view(1, 1, length) * torch.log(previous_timestep_weight)
        kernel = damping_factor * self.ema_expansion_matrix * torch.exp(vander)
        return torch.einsum('dnl,dn->dl', kernel, self.kernel_projection_matrix * self.scale)

    def get_ema_coefficients(self):
        if self.training:
            return self._compute_ema_coefficients()
        else:
            if self._coeffs is None:
                self._coeffs = self._compute_ema_coefficients()
            return self._coeffs

    def get_ema_kernel(self, length: int):
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        if self.training:
            return self._compute_efficient_ema_kernel(kernel_size)
        else:
            if self._kernel is None or self._kernel.size(-1) < kernel_size:
                self._kernel = self._compute_efficient_ema_kernel(kernel_size)
            return self._kernel[..., :kernel_size]

    def fft_convolution(self, inputs, kernel, length):
        inputs_fft = torch.fft.rfft(inputs.float(), n=2 * length)
        kernel_fft = torch.fft.rfft(kernel.float(), n=2 * length)
        convolved_sequence = torch.fft.irfft(inputs_fft * kernel_fft, n=2 * length)
        return convolved_sequence

    def ema_step(self, inputs, length, past_state=None):
        if length == 1:
            return self.one_ema_step(inputs, past_state=past_state)
        damping_factor, previous_timestep_weight = self.get_ema_coefficients()
        vander = torch.arange(length + 1).to(damping_factor).view(1, 1, length + 1) * torch.log(previous_timestep_weight)
        vander = torch.exp(vander)
        if past_state is not None:
            past_ema_proj = vander[:, :, 1:] * (self.kernel_projection_matrix * self.scale).unsqueeze(-1)
            past_ema_state = torch.einsum('bdn,dnl->bdl', past_state, past_ema_proj)
            past_vandermonde = vander[:, :, -1] * past_state
        else:
            past_ema_state = None
            past_vandermonde = None
        vander = vander[:, :, :-1]
        kernel = damping_factor * self.ema_expansion_matrix * vander
        kernel_proj = torch.einsum('dnl,dn->dl', kernel, self.kernel_projection_matrix * self.scale)
        ema_output = self.fft_convolution(inputs, kernel_proj, length=length)[..., 0:length]
        ema_output = ema_output.type_as(inputs)
        if past_ema_state is not None:
            ema_output = ema_output + past_ema_state
        updated_hidden_state = torch.einsum('bdl,dnl->bdn', inputs, torch.flip(kernel, dims=[2]))
        if past_vandermonde is not None:
            updated_hidden_state = updated_hidden_state + past_vandermonde
        return (ema_output.permute(2, 0, 1), updated_hidden_state)

    def one_ema_step(self, inputs, past_state=None):
        damping_factor, previous_timestep_weight = self.get_ema_coefficients()
        updated_state = (damping_factor * self.ema_expansion_matrix).squeeze(-1) * inputs
        if past_state is not None:
            updated_state = updated_state + previous_timestep_weight.squeeze(-1) * past_state
        out = torch.einsum('bdn,dn->bd', updated_state, self.kernel_projection_matrix * self.scale)
        return (out.unsqueeze(0), updated_state)

    def forward(self, inputs, attention_mask: Optional[torch.Tensor]=None, prev_state: Optional[torch.Tensor]=None, use_cache: bool=False) -> torch.Tensor:
        """
        Mega's exponential moving average (EMA) sub-layer applied prior to single-headed (traditional) self-attention

        Args:
            inputs (`torch.Tensor` of shape `(sequence_length, batch_size, hidden_size)`):
                Hidden state / embedding input to update via EMA based on FFT convolution
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indicates which inputs are to be ignored (mostly due to padding), where elements are either 1 for *not
                masked* or 0 for *masked*
            prev_state (`torch.Tensor` of shape `(batch_size, config.ndim)`, *optional*):
                The hidden state returned from the previous timestep during incremental decoding.
            use_cache (`bool`, default `False`):
                Whether to perfom incremental decoding; uses `prev_state` as the prior timestep, and returns the
                updated EMA hidden state for use in the next step

        Returns:
            `tuple(torch.FloatTensor)` containing various elements depending on configuration ([`MegaConfig`]) and
            inputs:
            - **hidden_states** (`torch.FloatTensor` of shape `(sequence_length, batch_size, hidden_size)`) -- Hidden
              states updated by EMA, with same shapes as inputs
            - **updated_state** (*optional*, returned when `use_cache=True`) `torch.FloatTensor of shape `(batch_size,
              config.ndim)` -- The incremental EMA state for use in the next step of incremental decoding
        """
        seq_len, bsz, embed_dim = inputs.size()
        if embed_dim != self.embed_dim:
            raise ValueError(f'Unexpected embedding dimension received: input is {embed_dim}, model expects {self.embed_dim}')
        residual = inputs * self.residual_weight
        inputs = inputs.permute(1, 2, 0)
        if attention_mask is not None:
            inputs = inputs * attention_mask.unsqueeze(1).type_as(inputs)
        if self.bidirectional and use_cache:
            raise RuntimeError('Bidirectional EMA does not support incremental state')
        if use_cache:
            out, updated_state = self.ema_step(inputs, seq_len, past_state=prev_state)
            out = F.silu(out + residual)
            return (out, updated_state)
        else:
            kernel = self.get_ema_kernel(seq_len)
            fft_len = seq_len
            s_index = 0
            kernel_size = kernel.size(1)
            if self.bidirectional:
                k1, k2 = torch.split(kernel, [self.embed_dim, self.embed_dim], dim=0)
                kernel = F.pad(k1, (kernel_size - 1, 0)) + F.pad(k2.flip(-1), (0, kernel_size - 1))
                inputs = F.pad(inputs, (kernel_size - 1, 0))
                fft_len = fft_len + kernel_size - 1
                s_index = 2 * kernel_size - 2
            ema_output = self.fft_convolution(inputs, kernel, length=fft_len)[..., s_index:s_index + seq_len]
            ema_output = ema_output.type_as(inputs)
            gated_ema_output = F.silu(ema_output.permute(2, 0, 1) + residual)
            return (gated_ema_output, None)