import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ....activations import ACT2FN
from ....file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ....integrations.deepspeed import is_deepspeed_zero3_enabled
from ....modeling_attn_mask_utils import _prepare_4d_attention_mask
from ....modeling_outputs import BaseModelOutput, CausalLMOutput
from ....modeling_utils import (
from ....utils import logging
from .configuration_mctct import MCTCTConfig
class MCTCTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MCTCTConfig
    base_model_prefix = 'mctct'
    main_input_name = 'input_features'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, MCTCTLayerNorm):
            module.singleton_weight.data.fill_(1.0)
            module.singleton_bias.data.zero_()
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        dilation = 1
        for _, kernel_sz, stride in zip(range(self.config.num_conv_layers), self.config.conv_kernel, self.config.conv_stride):
            padding = kernel_sz // 2
            input_lengths = input_lengths + 2 * padding - dilation * (kernel_sz - 1) - 1
            input_lengths = torch.div(input_lengths, stride, rounding_mode='trunc') + 1
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, -1]
        subsampled_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
        bsz = attention_mask.size()[0]
        attention_mask = torch.zeros((bsz, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask[torch.arange(bsz, device=attention_mask.device), subsampled_lengths - 1] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).long()
        return attention_mask