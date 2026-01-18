import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available
from .configuration_mamba import MambaConfig
from ..deprecated._archive_maps import MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: MambaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.conv1d = nn.Conv1d(in_channels=self.intermediate_size, out_channels=self.intermediate_size, bias=config.use_conv_bias, kernel_size=config.conv_kernel, groups=self.intermediate_size, padding=config.conv_kernel - 1)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias
        if not is_fast_path_available:
            logger.warning_once('The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)` is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and https://github.com/Dao-AILab/causal-conv1d')

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params: Optional[MambaCache]=None):
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        if self.training and cache_params is None:
            contextualized_states = mamba_inner_fn(projected_states, self.conv1d.weight, self.conv1d.bias if self.use_conv_bias else None, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias.float() if self.use_bias else None, -torch.exp(self.A_log.float()), None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        else:
            hidden_states, gate = projected_states.chunk(2, dim=1)
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if cache_params is not None and cache_params.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(hidden_states.squeeze(-1), cache_params.conv_states[self.layer_idx], conv_weights, self.conv1d.bias, self.activation)
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
                    cache_params.conv_states[self.layer_idx].copy_(conv_states)
                hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv1d.bias, activation=self.activation)
            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
            discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)
            A = -torch.exp(self.A_log.float())
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, 'bias') else None
            if cache_params is not None and cache_params.seqlen_offset > 0:
                scan_outputs = selective_state_update(cache_params.ssm_states[self.layer_idx], hidden_states[..., 0], discrete_time_step[..., 0], A, B[:, 0], C[:, 0], self.D, gate[..., 0], time_proj_bias, dt_softplus=True).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(hidden_states, discrete_time_step, A, B.transpose(1, 2), C.transpose(1, 2), self.D.float(), gate, time_proj_bias, delta_softplus=True, return_last_state=True)
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
            contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    def slow_forward(self, input_states, cache_params: Optional[MambaCache]=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        projected_states = self.in_proj(input_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)
            else:
                conv_state = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
        else:
            ssm_state = torch.zeros((batch_size, self.intermediate_size, self.ssm_state_size), device=hidden_states.device, dtype=dtype)
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
        discrete_time_step = self.dt_proj(time_step)
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)
        scan_output = scan_output + hidden_states * self.D[None, :, None]
        scan_output = scan_output * self.act(gate)
        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))
        return contextualized_states

    def forward(self, hidden_states, cache_params: Optional[MambaCache]=None):
        if is_fast_path_available and 'cuda' in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params)
        return self.slow_forward(hidden_states, cache_params)