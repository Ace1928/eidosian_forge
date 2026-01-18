import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ibert import IBertConfig
from .quant_modules import IntGELU, IntLayerNorm, IntSoftmax, QuantAct, QuantEmbedding, QuantLinear
class IBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = IBertConfig
    base_model_prefix = 'ibert'

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (QuantLinear, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (QuantEmbedding, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (IntLayerNorm, nn.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError('`resize_token_embeddings` is not supported for I-BERT.')