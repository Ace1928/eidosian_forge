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
class BridgeTowerMLMHead(nn.Module):

    def __init__(self, config, weight=None):
        super().__init__()
        self.config = config
        self.transform = BridgeTowerPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.text_config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.text_config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        mlm_score = self.transform(x)
        mlm_score = self.decoder(mlm_score) + self.bias
        return mlm_score