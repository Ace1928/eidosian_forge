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
def _init_weight(self, weight):
    if self.config.init == 'uniform':
        nn.init.uniform_(weight, -self.config.init_range, self.config.init_range)
    elif self.config.init == 'normal':
        nn.init.normal_(weight, 0.0, self.config.init_std)