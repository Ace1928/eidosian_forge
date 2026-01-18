import collections
import math
from typing import Iterable, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_pvt import PvtConfig
class PvtPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = PvtConfig
    base_model_prefix = 'pvt'
    main_input_name = 'pixel_values'

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PvtPatchEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(module.position_embeddings.data, mean=0.0, std=self.config.initializer_range)
            if module.cls_token is not None:
                module.cls_token.data = nn.init.trunc_normal_(module.cls_token.data, mean=0.0, std=self.config.initializer_range)