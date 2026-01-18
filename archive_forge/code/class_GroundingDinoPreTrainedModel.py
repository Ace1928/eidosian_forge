import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_grounding_dino import GroundingDinoConfig
class GroundingDinoPreTrainedModel(PreTrainedModel):
    config_class = GroundingDinoConfig
    base_model_prefix = 'model'
    main_input_name = 'pixel_values'

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, GroundingDinoLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, GroundingDinoMultiscaleDeformableAttention):
            module._reset_parameters()
        elif isinstance(module, GroundingDinoBiMultiHeadAttention):
            nn.init.xavier_uniform_(module.vision_proj.weight)
            module.vision_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.text_proj.weight)
            module.text_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.values_vision_proj.weight)
            module.values_vision_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.values_text_proj.weight)
            module.values_text_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.out_vision_proj.weight)
            module.out_vision_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.out_text_proj.weight)
            module.out_text_proj.bias.data.fill_(0)
        elif isinstance(module, (GroundingDinoEncoderLayer, GroundingDinoDecoderLayer)):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0.0, std=std)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GroundingDinoMLPPredictionHead):
            nn.init.constant_(module.layers[-1].weight.data, 0)
            nn.init.constant_(module.layers[-1].bias.data, 0)
        if hasattr(module, 'reference_points') and (not self.config.two_stage):
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        if hasattr(module, 'level_embed'):
            nn.init.normal_(module.level_embed)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GroundingDinoDecoder):
            module.gradient_checkpointing = value