import warnings
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch import Tensor
from torch.masked import as_masked_tensor, is_masked_tensor, MaskedTensor
from . import _docs
from torch._prims_common import corresponding_real_dtype
from torch import sym_float
def _sparse_csr_where(mask: Tensor, input: Tensor, fill_value: Tensor) -> Tensor:
    """Sparse variant of torch.where. Supports sparse CSR tensors."""
    return _sparse_coo_where(mask.to_sparse_coo(), input.to_sparse_coo(), fill_value).to_sparse_csr()