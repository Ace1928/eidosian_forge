import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
def _append_cache(self, key, value):
    if 'key' not in self.cache:
        self.cache['key'] = key
        self.cache['value'] = value
    else:
        old_key, old_value = (key, value)
        key = torch.cat([self.cache['key'], old_key], dim=1)
        value = torch.cat([self.cache['value'], old_value], dim=1)
        del self.cache['key']
        del self.cache['value']
        del old_key
        del old_value
        self.cache['key'] = key
        self.cache['value'] = value
    return (self.cache['key'], self.cache['value'])