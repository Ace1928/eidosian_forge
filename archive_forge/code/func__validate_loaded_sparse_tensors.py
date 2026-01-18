import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _validate_loaded_sparse_tensors():
    try:
        for t in _sparse_tensors_to_validate:
            if t.layout is torch.sparse_coo:
                torch._validate_sparse_coo_tensor_args(t._indices(), t._values(), t.size(), t.is_coalesced())
            elif t.layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
                if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                    compressed_indices, plain_indices = (t.crow_indices(), t.col_indices())
                else:
                    compressed_indices, plain_indices = (t.ccol_indices(), t.row_indices())
                torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, t.values(), t.size(), t.layout)
            else:
                raise NotImplementedError(f'_validate_loaded_sparse_tensors for layout `{t.layout}`')
    finally:
        _sparse_tensors_to_validate.clear()