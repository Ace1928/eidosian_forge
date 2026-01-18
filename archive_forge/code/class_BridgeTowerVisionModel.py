import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig
class BridgeTowerVisionModel(BridgeTowerPreTrainedModel):
    config_class = BridgeTowerVisionConfig

    def __init__(self, config):
        super().__init__(config)
        self.visual = BridgeTowerVisionTransformer(config)

    @property
    def dtype(self):
        return self.visual.embeddings.patch_embedding.weight.dtype

    def forward(self, image, image_mask=None):
        return self.visual(image.type(self.dtype), image_mask)