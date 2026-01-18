from enum import Enum
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers import _is_triton_available
from collections import namedtuple
class ResidualNormStyle(str, Enum):
    """Support different residual path and norm styles.
    See "On Layer Normalization in the Transformer Architecture",
    Xiong et al., https://arxiv.org/pdf/2002.04745v1.pdf
    """
    Pre = 'pre'
    Post = 'post'
    DeepNorm = 'deepnorm'