import math
from typing import List, Optional, Tuple
import torch
from torchaudio.models.emformer import _EmformerAttention, _EmformerImpl, _get_weight_init_gains
class _ConvolutionModule(torch.nn.Module):

    def __init__(self, input_dim: int, segment_length: int, right_context_length: int, kernel_size: int, activation: str='silu', dropout: float=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.segment_length = segment_length
        self.right_context_length = right_context_length
        self.state_size = kernel_size - 1
        self.pre_conv = torch.nn.Sequential(torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, 2 * input_dim, bias=True), torch.nn.GLU())
        self.conv = torch.nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=kernel_size, stride=1, padding=0, groups=input_dim)
        self.post_conv = torch.nn.Sequential(torch.nn.LayerNorm(input_dim), _get_activation_module(activation), torch.nn.Linear(input_dim, input_dim, bias=True), torch.nn.Dropout(p=dropout))

    def _split_right_context(self, utterance: torch.Tensor, right_context: torch.Tensor) -> torch.Tensor:
        T, B, D = right_context.size()
        if T % self.right_context_length != 0:
            raise ValueError('Tensor length should be divisible by its right context length')
        num_segments = T // self.right_context_length
        right_context_segments = right_context.reshape(num_segments, self.right_context_length, B, D)
        right_context_segments = right_context_segments.permute(0, 2, 1, 3).reshape(num_segments * B, self.right_context_length, D)
        pad_segments = []
        for seg_idx in range(num_segments):
            end_idx = min(self.state_size + (seg_idx + 1) * self.segment_length, utterance.size(0))
            start_idx = end_idx - self.state_size
            pad_segments.append(utterance[start_idx:end_idx, :, :])
        pad_segments = torch.cat(pad_segments, dim=1).permute(1, 0, 2)
        return torch.cat([pad_segments, right_context_segments], dim=1).permute(0, 2, 1)

    def _merge_right_context(self, right_context: torch.Tensor, B: int) -> torch.Tensor:
        right_context = right_context.reshape(-1, B, self.input_dim, self.right_context_length)
        right_context = right_context.permute(0, 3, 1, 2)
        return right_context.reshape(-1, B, self.input_dim)

    def forward(self, utterance: torch.Tensor, right_context: torch.Tensor, state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input = torch.cat((right_context, utterance))
        x = self.pre_conv(input)
        x_right_context, x_utterance = (x[:right_context.size(0), :, :], x[right_context.size(0):, :, :])
        x_utterance = x_utterance.permute(1, 2, 0)
        if state is None:
            state = torch.zeros(input.size(1), input.size(2), self.state_size, device=input.device, dtype=input.dtype)
        state_x_utterance = torch.cat([state, x_utterance], dim=2)
        conv_utterance = self.conv(state_x_utterance)
        conv_utterance = conv_utterance.permute(2, 0, 1)
        if self.right_context_length > 0:
            right_context_block = self._split_right_context(state_x_utterance.permute(2, 0, 1), x_right_context)
            conv_right_context_block = self.conv(right_context_block)
            conv_right_context = self._merge_right_context(conv_right_context_block, input.size(1))
            y = torch.cat([conv_right_context, conv_utterance], dim=0)
        else:
            y = conv_utterance
        output = self.post_conv(y) + input
        new_state = state_x_utterance[:, :, -self.state_size:]
        return (output[right_context.size(0):], output[:right_context.size(0)], new_state)

    def infer(self, utterance: torch.Tensor, right_context: torch.Tensor, state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input = torch.cat((utterance, right_context))
        x = self.pre_conv(input)
        x = x.permute(1, 2, 0)
        if state is None:
            state = torch.zeros(input.size(1), input.size(2), self.state_size, device=input.device, dtype=input.dtype)
        state_x = torch.cat([state, x], dim=2)
        conv_out = self.conv(state_x)
        conv_out = conv_out.permute(2, 0, 1)
        output = self.post_conv(conv_out) + input
        new_state = state_x[:, :, -self.state_size - right_context.size(0):-right_context.size(0)]
        return (output[:utterance.size(0)], output[utterance.size(0):], new_state)