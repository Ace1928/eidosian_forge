import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
@torch.jit.unused
def _get_aux_predictions(self, outputs_class, outputs_seg_masks):
    aux_list = [{'class_queries_logits': a, 'masks_queries_logits': b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
    return tuple(aux_list)