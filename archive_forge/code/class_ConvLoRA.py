import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
class ConvLoRA(nn.Module, LoRALayer):

    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0.0, merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        if r > 0:
            self.lora_A = nn.Parameter(self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.conv.weight.new_zeros((out_channels // self.conv.groups * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        elif self.merge_weights and (not self.merged):
            if self.r > 0:
                self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x):
        if self.r > 0 and (not self.merged):
            return self.conv._conv_forward(x, self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling, self.conv.bias)
        return self.conv(x)