import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_wavlm import WavLMConfig
def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
    """
        Computes the output length of the TDNN layers
        """

    def _conv_out_length(input_length, kernel_size, stride):
        return (input_length - kernel_size) // stride + 1
    for kernel_size in self.config.tdnn_kernel:
        input_lengths = _conv_out_length(input_lengths, kernel_size, 1)
    return input_lengths