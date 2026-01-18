import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
class Linear8bit(nn.Module):

    def __init__(self, input_features, output_features, bias=True, args=None):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.args = args
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        self.args.training = self.training
        return LinearFunction.apply(x, self.weight, self.bias, self.args)