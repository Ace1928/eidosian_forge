import re
from typing import Callable, Dict, Optional, Set, Union
import torch.fx
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module
def _create_param(i):
    return torch.nn.Parameter(i if not isinstance(i, int) else torch.Tensor([i]).to(device=self.device_for_folded_attrs), requires_grad=i.requires_grad if isinstance(i, torch.Tensor) else False)