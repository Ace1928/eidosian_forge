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
def _resize_qa_labels(self, num_labels):
    cur_qa_logit_layer = self.get_qa_logit_layer()
    new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
    self._set_qa_logit_layer(new_qa_logit_layer)
    return self.get_qa_logit_layer()