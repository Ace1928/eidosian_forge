from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import ModelOutput, PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_univnet import UnivNetConfig
class UnivNetKernelPredictor(nn.Module):
    """
    Implementation of the kernel predictor network which supplies the kernel and bias for the location variable
    convolutional layers (LVCs) in each UnivNet LVCBlock.

    Based on the KernelPredictor implementation in
    [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/model/lvcnet.py#L7).

    Parameters:
        config: (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
        conv_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the location variable convolutional layer kernels (convolutional weight tensor).
        conv_layers (`int`, *optional*, defaults to 4):
            The number of location variable convolutional layers to output kernels and biases for.
    """

    def __init__(self, config: UnivNetConfig, conv_kernel_size: int=3, conv_layers: int=4):
        super().__init__()
        self.conv_in_channels = config.model_hidden_channels
        self.conv_out_channels = 2 * config.model_hidden_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers
        self.kernel_channels = self.conv_in_channels * self.conv_out_channels * self.conv_kernel_size * self.conv_layers
        self.bias_channels = self.conv_out_channels * self.conv_layers
        self.resnet_in_channels = config.num_mel_bins
        self.resnet_hidden_channels = config.kernel_predictor_hidden_channels
        self.resnet_kernel_size = config.kernel_predictor_conv_size
        self.num_blocks = config.kernel_predictor_num_blocks
        self.leaky_relu_slope = config.leaky_relu_slope
        padding = (self.resnet_kernel_size - 1) // 2
        self.input_conv = nn.Conv1d(self.resnet_in_channels, self.resnet_hidden_channels, 5, padding=2, bias=True)
        self.resblocks = nn.ModuleList([UnivNetKernelPredictorResidualBlock(config) for _ in range(self.num_blocks)])
        self.kernel_conv = nn.Conv1d(self.resnet_hidden_channels, self.kernel_channels, self.resnet_kernel_size, padding=padding, bias=True)
        self.bias_conv = nn.Conv1d(self.resnet_hidden_channels, self.bias_channels, self.resnet_kernel_size, padding=padding, bias=True)

    def forward(self, spectrogram: torch.FloatTensor):
        """
        Maps a conditioning log-mel spectrogram to a tensor of convolutional kernels and biases, for use in location
        variable convolutional layers. Note that the input spectrogram should have shape (batch_size, input_channels,
        seq_length).

        Args:
            spectrogram (`torch.FloatTensor` of shape `(batch_size, input_channels, seq_length)`):
                Tensor containing the log-mel spectrograms.

        Returns:
            Tuple[`torch.FloatTensor, `torch.FloatTensor`]: tuple of tensors where the first element is the tensor of
            location variable convolution kernels of shape `(batch_size, self.conv_layers, self.conv_in_channels,
            self.conv_out_channels, self.conv_kernel_size, seq_length)` and the second element is the tensor of
            location variable convolution biases of shape `(batch_size, self.conv_layers. self.conv_out_channels,
            seq_length)`.
        """
        batch_size, _, seq_length = spectrogram.shape
        hidden_states = self.input_conv(spectrogram)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        for resblock in self.resblocks:
            hidden_states = resblock(hidden_states)
        kernel_hidden_states = self.kernel_conv(hidden_states)
        bias_hidden_states = self.bias_conv(hidden_states)
        kernels = kernel_hidden_states.view(batch_size, self.conv_layers, self.conv_in_channels, self.conv_out_channels, self.conv_kernel_size, seq_length).contiguous()
        biases = bias_hidden_states.view(batch_size, self.conv_layers, self.conv_out_channels, seq_length).contiguous()
        return (kernels, biases)

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.input_conv)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.kernel_conv)
        nn.utils.weight_norm(self.bias_conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.input_conv)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.kernel_conv)
        nn.utils.remove_weight_norm(self.bias_conv)