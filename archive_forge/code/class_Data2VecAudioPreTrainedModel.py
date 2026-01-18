import math
import warnings
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
from .configuration_data2vec_audio import Data2VecAudioConfig
class Data2VecAudioPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Data2VecAudioConfig
    base_model_prefix = 'data2vec_audio'
    main_input_name = 'input_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Data2VecAudioFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, Data2VecAudioPositionalConvLayer):
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool]=None):
        """
        Computes the output length of the convolutional layers
        """
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode='floor') + 1
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None):
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)
        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask[torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask