import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seggpt import SegGptConfig
from ..deprecated._archive_maps import SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
class SegGptPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = SegGptConfig
    base_model_prefix = 'model'
    main_input_name = 'pixel_values'
    supports_gradient_checkpointing = True
    _no_split_modules = ['SegGptEmbeddings', 'SegGptLayer']

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data.to(torch.float32), mean=0.0, std=std).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, SegGptAttention):
            module.rel_pos_h.data = nn.init.trunc_normal_(module.rel_pos_h.data.to(torch.float32), mean=0.0, std=std).to(module.rel_pos_h.dtype)
            module.rel_pos_w.data = nn.init.trunc_normal_(module.rel_pos_w.data.to(torch.float32), mean=0.0, std=std).to(module.rel_pos_w.dtype)
        elif isinstance(module, SegGptEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(module.position_embeddings.data.to(torch.float32), mean=0.0, std=std).to(module.position_embeddings.dtype)
            torch.nn.init.normal_(module.mask_token, std=std)
            torch.nn.init.normal_(module.segment_token_input, std=std)
            torch.nn.init.normal_(module.segment_token_prompt, std=std)
            torch.nn.init.normal_(module.type_token_semantic, std=std)
            torch.nn.init.normal_(module.type_token_instance, std=std)