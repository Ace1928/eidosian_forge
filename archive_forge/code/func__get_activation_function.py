from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from .base import BetterTransformerBaseLayer
def _get_activation_function(self, config: 'PretrainedConfig'):
    if hasattr(config, 'vision_config') and hasattr(config, 'text_config'):
        assert config.vision_config.hidden_act == config.text_config.hidden_act
        return config.vision_config.hidden_act
    else:
        return config.hidden_act