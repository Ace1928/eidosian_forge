import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
class BigBirdForQuestionAnsweringHead(nn.Module):
    """Head for question answering tasks."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intermediate = BigBirdIntermediate(config)
        self.output = BigBirdOutput(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_output):
        hidden_states = self.dropout(encoder_output)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states, encoder_output)
        hidden_states = self.qa_outputs(hidden_states)
        return hidden_states