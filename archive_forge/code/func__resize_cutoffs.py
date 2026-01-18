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
def _resize_cutoffs(self, new_num_tokens, new_emb_size, new_embedding_shapes, layer):
    new_cutoffs = super()._resize_cutoffs(new_num_tokens, new_emb_size, new_embedding_shapes, layer)
    self.crit.cutoffs = new_cutoffs
    self.crit.cutoff_ends = [0] + new_cutoffs
    self.crit.n_token = new_num_tokens