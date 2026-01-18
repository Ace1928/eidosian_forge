import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_unispeech_sat import UniSpeechSatConfig
class UniSpeechSatPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = UniSpeechSatConfig
    base_model_prefix = 'unispeech_sat'
    main_input_name = 'input_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, UniSpeechSatGumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        elif isinstance(module, UniSpeechSatPositionalConvEmbedding):
            nn.init.normal_(module.conv.weight, mean=0, std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)))
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, UniSpeechSatFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode='floor') + 1
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths).to(torch.long)
        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask[torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask