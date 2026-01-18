import torch
from enum import Enum
from torch._C import _MobileOptimizerType as MobileOptimizerType
from typing import Optional, Set, List, AnyStr
class LintCode(Enum):
    BUNDLED_INPUT = 1
    REQUIRES_GRAD = 2
    DROPOUT = 3
    BATCHNORM = 4