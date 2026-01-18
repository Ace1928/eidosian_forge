import math
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import neginf, PipelineHelper
class TransformerLinearWrapper(nn.Module):
    """
    Wrap a transformer in a linear layer.
    """

    def __init__(self, transformer, output_dim):
        super().__init__()
        self.transformer = transformer
        input_dim = transformer.out_dim
        self.additional_linear_layer = nn.Linear(input_dim, output_dim)

    def forward(self, *args):
        """
        Forward pass.

        Apply transformer, then additional linear layer.
        """
        context_h = self.transformer(*args)
        return self.additional_linear_layer(context_h)