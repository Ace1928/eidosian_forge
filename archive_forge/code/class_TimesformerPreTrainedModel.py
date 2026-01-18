import collections
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_timesformer import TimesformerConfig
class TimesformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = TimesformerConfig
    base_model_prefix = 'timesformer'
    main_input_name = 'pixel_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, TimesformerEmbeddings):
            nn.init.trunc_normal_(module.cls_token, std=self.config.initializer_range)
            nn.init.trunc_normal_(module.position_embeddings, std=self.config.initializer_range)
            module.patch_embeddings.apply(self._init_weights)