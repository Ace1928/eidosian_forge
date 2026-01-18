import itertools
import torch
from torch.utils import benchmark
from xformers.components.attention._sputnik_sparse import _csr_to_coo
from xformers.components.attention.core import SparseCS, _create_random_sparsity
def _get_fn(backend):
    if backend == 'csr_ge':
        fn = torch.ops.xformers.csr_sddmm
    elif backend == 'csr_sputnik':
        fn = torch.ops.xformers.sddmm_sputnik
    elif backend == 'coo_ge':

        def fn(a, b, row_indices, row_offsets, column_indices):
            row_coo, _ = _csr_to_coo(a.shape[-2], b.shape[-2], row_offsets, column_indices)
            return torch.ops.xformers.coo_sddmm(a, b, row_indices, row_coo, column_indices)
    elif backend == 'csr_to_coo':

        def fn(a, b, row_indices, row_offsets, column_indices):
            row_coo, _ = _csr_to_coo(a.shape[-2], b.shape[-2], row_offsets, column_indices)
            return row_coo
    return fn