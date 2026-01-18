from typing import (
import operator
import itertools
import torch
from torch._C import _add_docstr
import torch.nn.functional as F
from ._lowrank import svd_lowrank, pca_lowrank
from .overrides import (
from ._jit_internal import boolean_dispatch
from ._jit_internal import _overload as overload
from torch import _VF
def _meshgrid(*tensors, indexing: Optional[str]):
    if has_torch_function(tensors):
        return handle_torch_function(meshgrid, tensors, *tensors, indexing=indexing)
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tensors[0]
    kwargs = {} if indexing is None else {'indexing': indexing}
    return _VF.meshgrid(tensors, **kwargs)