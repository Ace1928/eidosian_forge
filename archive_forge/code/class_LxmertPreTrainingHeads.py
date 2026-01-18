import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_lxmert import LxmertConfig
class LxmertPreTrainingHeads(nn.Module):

    def __init__(self, config, lxmert_model_embedding_weights):
        super(LxmertPreTrainingHeads, self).__init__()
        self.predictions = LxmertLMPredictionHead(config, lxmert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return (prediction_scores, seq_relationship_score)