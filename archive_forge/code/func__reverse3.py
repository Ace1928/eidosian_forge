import torch.nn as nn
import torch.ao.nn.intrinsic as nni
from typing import Any, Union, Callable, List, Tuple, Dict, Optional, Type
from torch.ao.quantization.utils import Pattern, get_combined_dict, MatchAllNode
import itertools
def _reverse3(f):

    def reversed(is_qat, x, w):
        y, z = w
        return f(is_qat, z, y, x)
    return reversed