import inspect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...utils import (
from ..auto import AutoModel
from .configuration_idefics2 import Idefics2Config, Idefics2VisionConfig
class Idefics2Connector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.modality_projection = Idefics2MLP(hidden_size=config.vision_config.hidden_size, intermediate_size=config.text_config.intermediate_size, output_size=config.text_config.hidden_size, hidden_act=config.text_config.hidden_act)
        self.perceiver_resampler = Idefics2PerceiverResampler(config)

    def forward(self, image_hidden_states, attention_mask):
        image_hidden_states = self.modality_projection(image_hidden_states)
        image_hidden_states = self.perceiver_resampler(context=image_hidden_states, attention_mask=attention_mask)
        return image_hidden_states