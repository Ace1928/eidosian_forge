import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
def _init_residue_constants(self, float_dtype, device):
    if not hasattr(self, 'default_frames'):
        self.register_buffer('default_frames', torch.tensor(residue_constants.restype_rigid_group_default_frame, dtype=float_dtype, device=device, requires_grad=False), persistent=False)
    if not hasattr(self, 'group_idx'):
        self.register_buffer('group_idx', torch.tensor(residue_constants.restype_atom14_to_rigid_group, device=device, requires_grad=False), persistent=False)
    if not hasattr(self, 'atom_mask'):
        self.register_buffer('atom_mask', torch.tensor(residue_constants.restype_atom14_mask, dtype=float_dtype, device=device, requires_grad=False), persistent=False)
    if not hasattr(self, 'lit_positions'):
        self.register_buffer('lit_positions', torch.tensor(residue_constants.restype_atom14_rigid_group_positions, dtype=float_dtype, device=device, requires_grad=False), persistent=False)