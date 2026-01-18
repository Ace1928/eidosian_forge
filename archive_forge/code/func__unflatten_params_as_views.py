from contextlib import contextmanager
from itertools import chain
import typing
from typing import (
import torch
from torch import Tensor
import torch.nn as nn
from fairscale.internal.state_dict import replace_by_prefix_
def _unflatten_params_as_views(self) -> None:
    """Unlike ``_unflatten_params``, this function unflatten into views and keep
        self.flat_param unchanged.
        """
    assert self.is_flattened
    ps = self.get_param_views()
    param_views = []
    for (_, m, n), p in zip(self._param_infos, ps):
        setattr(m, n, p)
        param_views.append(p)
    setattr(self._fpw_module, '_unflattened_param_views', param_views)
    for _, _, m, n, shared_m, shared_n in self._shared_param_infos:
        setattr(m, n, getattr(shared_m, shared_n))