import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
class FastSpeech2ConformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FastSpeech2ConformerConfig
    base_model_prefix = 'fastspeech2_conformer'
    main_input_name = 'input_ids'

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                key = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-key, b=key)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_()
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, FastSpeech2ConformerAttention):
            nn.init.xavier_uniform_(module.pos_bias_u)
            nn.init.xavier_uniform_(module.pos_bias_v)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FastSpeech2ConformerEncoder):
            module.gradient_checkpointing = value