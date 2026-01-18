import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....modeling_utils import PreTrainedModel
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_transfo_xl_utilities import ProjectedAdaptiveLogSoftmax
def _init_bias(self, bias):
    nn.init.constant_(bias, 0.0)