from contextlib import contextmanager
from itertools import chain
import typing
from typing import (
import torch
from torch import Tensor
import torch.nn as nn
from fairscale.internal.state_dict import replace_by_prefix_
@property
def _param_infos(self) -> Iterator[Tuple[str, nn.Module, str]]:
    return chain(*[p._param_infos for p in self.flat_params])