import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...cache_utils import DynamicCache  # we need __iter__ and __len__ of pkv
from ...modeling_attn_mask_utils import (
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.import_utils import (
from .configuration_jamba import JambaConfig
class JambaMambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: JambaConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.conv1d = nn.Conv1d(in_channels=self.intermediate_size, out_channels=self.intermediate_size, bias=self.use_conv_bias, kernel_size=self.conv_kernel_size, groups=self.intermediate_size, padding=self.conv_kernel_size - 1)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.use_fast_kernels = config.use_mamba_kernels
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)
        self.dt_layernorm = JambaRMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
        self.b_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        self.c_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        if not is_fast_path_available:
            logger.warning_once('The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)` is None. To install follow https://github.com/state-spaces/mamba/#installation and https://github.com/Dao-AILab/causal-conv1d. If you want to use the naive implementation, set `use_mamba_kernels=False` in the model config')

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params: HybridMambaAttentionDynamicCache=None):
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state and (seq_len == 1) and (cache_params.conv_states[self.layer_idx].shape[0] == cache_params.ssm_states[self.layer_idx].shape[0] == batch_size)
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if use_precomputed_states:
            hidden_states = causal_conv1d_update(hidden_states.squeeze(-1), cache_params.conv_states[self.layer_idx], conv_weights, self.conv1d.bias, self.activation)
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            if cache_params is not None:
                conv_states = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
                cache_params.conv_states[self.layer_idx].copy_(conv_states)
            hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv1d.bias, activation=self.activation)
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)
        time_proj_bias = self.dt_proj.bias
        self.dt_proj.bias = None
        discrete_time_step = self.dt_proj(time_step).transpose(1, 2)
        self.dt_proj.bias = time_proj_bias
        A = -torch.exp(self.A_log.float())
        time_proj_bias = time_proj_bias.float() if time_proj_bias is not None else None
        if use_precomputed_states:
            scan_outputs = selective_state_update(cache_params.ssm_states[self.layer_idx], hidden_states[..., 0], discrete_time_step[..., 0], A, B[:, 0], C[:, 0], self.D, gate[..., 0], time_proj_bias, dt_softplus=True).unsqueeze(-1)
        else:
            scan_outputs, ssm_state = selective_scan_fn(hidden_states, discrete_time_step, A, B.transpose(1, 2), C.transpose(1, 2), self.D.float(), gate, time_proj_bias, delta_softplus=True, return_last_state=True)
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    def slow_forward(self, input_states, cache_params: HybridMambaAttentionDynamicCache=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        projected_states = self.in_proj(input_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)
        use_cache = isinstance(cache_params, HybridMambaAttentionDynamicCache)
        if use_cache and cache_params.ssm_states[self.layer_idx].shape[0] == batch_size:
            if self.training:
                ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            else:
                ssm_state = cache_params.ssm_states[self.layer_idx]
            if cache_params.has_previous_state and seq_len == 1 and (cache_params.conv_states[self.layer_idx].shape[0] == batch_size):
                conv_state = cache_params.conv_states[self.layer_idx]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)
            else:
                conv_state = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
        else:
            ssm_state = torch.zeros((batch_size, self.intermediate_size, self.ssm_state_size), device=hidden_states.device, dtype=dtype)
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)
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
        if use_cache:
            cache_params.ssm_states[self.layer_idx] = ssm_state
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))
        return contextualized_states

    def forward(self, hidden_states, cache_params: HybridMambaAttentionDynamicCache=None):
        if self.use_fast_kernels:
            if not is_fast_path_available or 'cuda' not in self.x_proj.weight.device.type:
                raise ValueError('Fast Mamba kernels are not available. Make sure to they are installed and that the mamba module is on a CUDA device')
            return self.cuda_kernels_forward(hidden_states, cache_params)
        return self.slow_forward(hidden_states, cache_params)