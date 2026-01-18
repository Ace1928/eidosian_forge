import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...utils import is_scipy_available
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_fnet import FNetConfig
class FNetOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = FNetLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores