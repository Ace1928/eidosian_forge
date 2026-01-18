from enum import Enum
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers import _is_triton_available
from collections import namedtuple
class NormalizationType(str, Enum):
    LayerNorm = 'layernorm'
    Skip = 'skip'