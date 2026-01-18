import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig
class VitsHifiGan(nn.Module):

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d(config.flow_size, config.upsample_initial_channel, kernel_size=7, stride=1, padding=3)
        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(nn.ConvTranspose1d(config.upsample_initial_channel // 2 ** i, config.upsample_initial_channel // 2 ** (i + 1), kernel_size=kernel_size, stride=upsample_rate, padding=(kernel_size - upsample_rate) // 2))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // 2 ** (i + 1)
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))
        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3, bias=False)
        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.upsample_initial_channel, 1)

    def apply_weight_norm(self):
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()

    def remove_weight_norm(self):
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()

    def forward(self, spectrogram: torch.FloatTensor, global_conditioning: Optional[torch.FloatTensor]=None) -> torch.FloatTensor:
        """
        Converts a spectrogram into a speech waveform.

        Args:
            spectrogram (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`):
                Tensor containing the spectrograms.
            global_conditioning (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_size, 1)`, *optional*):
                Tensor containing speaker embeddings, for multispeaker models.

        Returns:
            `torch.FloatTensor`: Tensor of shape shape `(batch_size, 1, num_frames)` containing the speech waveform.
        """
        hidden_states = self.conv_pre(spectrogram)
        if global_conditioning is not None:
            hidden_states = hidden_states + self.cond(global_conditioning)
        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)
            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels
        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        waveform = torch.tanh(hidden_states)
        return waveform