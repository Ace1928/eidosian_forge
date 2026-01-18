import abc
import torch
from typing import Optional, Tuple, List, Any, Dict
from ...sparsifier import base_sparsifier
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier import utils
from torch.nn.utils import parametrize
import sys
import warnings
def _extract_weight(self, data):
    if type(data) in [torch.Tensor, nn.Parameter]:
        return data
    elif type(data) in EMBEDDING_TYPES:
        return data.weight