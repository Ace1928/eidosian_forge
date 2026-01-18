from contextlib import contextmanager
from itertools import chain
import typing
from typing import (
import torch
from torch import Tensor
import torch.nn as nn
from fairscale.internal.state_dict import replace_by_prefix_
@contextmanager
def _no_auto_unflatten_state_dict(self) -> Generator:
    backup = self._auto_unflatten_state_dict
    self._auto_unflatten_state_dict = False
    try:
        yield
    finally:
        self._auto_unflatten_state_dict = backup