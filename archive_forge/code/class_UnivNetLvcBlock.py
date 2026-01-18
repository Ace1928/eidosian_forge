from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import ModelOutput, PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_univnet import UnivNetConfig
class UnivNetLvcBlock(nn.Module):
    """
    Implementation of the location variable convolution (LVC) residual block of the UnivNet residual block. Includes a
    `UnivNetKernelPredictor` inside to predict the kernels and biases of the LVC layers.

    Based on LVCBlock in
    [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/model/lvcnet.py#L98)

    Parameters:
        config (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
        layer_id (`int`):
            An integer corresponding to the index of the current LVC resnet block layer. This should be between 0 and
            `len(config.resblock_stride_sizes) - 1)` inclusive.
        lvc_hop_size (`int`, *optional*, defaults to 256):
            The hop size for the location variable convolutional layers.
    """

    def __init__(self, config: UnivNetConfig, layer_id: int, lvc_hop_size: int=256):
        super().__init__()
        self.hidden_channels = config.model_hidden_channels
        self.kernel_size = config.resblock_kernel_sizes[layer_id]
        self.stride = config.resblock_stride_sizes[layer_id]
        self.dilations = config.resblock_dilation_sizes[layer_id]
        self.cond_hop_length = lvc_hop_size
        self.leaky_relu_slope = config.leaky_relu_slope
        self.num_blocks = len(self.dilations)
        self.convt_pre = nn.ConvTranspose1d(self.hidden_channels, self.hidden_channels, 2 * self.stride, stride=self.stride, padding=self.stride // 2 + self.stride % 2, output_padding=self.stride % 2)
        self.kernel_predictor = UnivNetKernelPredictor(config, self.kernel_size, self.num_blocks)
        self.resblocks = nn.ModuleList([UnivNetLvcResidualBlock(config, self.kernel_size, self.dilations[i]) for i in range(self.num_blocks)])

    def forward(self, hidden_states: torch.FloatTensor, spectrogram: torch.FloatTensor):
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.convt_pre(hidden_states)
        kernels, biases = self.kernel_predictor(spectrogram)
        for i, resblock in enumerate(self.resblocks):
            kernel = kernels[:, i, :, :, :, :]
            bias = biases[:, i, :, :]
            hidden_states = resblock(hidden_states, kernel, bias, hop_size=self.cond_hop_length)
        return hidden_states

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.convt_pre)
        self.kernel_predictor.apply_weight_norm()
        for layer in self.resblocks:
            layer.apply_weight_norm()

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.convt_pre)
        self.kernel_predictor.remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()