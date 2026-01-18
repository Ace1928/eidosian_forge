import warnings
import torch
from torch.overrides import get_default_nowrap_functions
def _validate_members(self):
    data = self._masked_data
    mask = self.get_mask()
    if type(data) != type(mask):
        raise TypeError(f'data and mask must have the same type. Got {type(data)} and {type(mask)}')
    if data.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr}:
        raise TypeError(f'data layout of {data.layout} is not supported.')
    if data.layout == torch.sparse_coo:
        if not _tensors_match(data.indices(), mask.indices(), exact=True):
            raise ValueError('data and mask are both sparse COO tensors but do not have the same indices.')
    elif data.layout == torch.sparse_csr:
        if not _tensors_match(data.crow_indices(), mask.crow_indices(), exact=True) or not _tensors_match(data.col_indices(), mask.col_indices(), exact=True):
            raise ValueError('data and mask are both sparse CSR tensors but do not share either crow or col indices.')
    if mask.dtype != torch.bool:
        raise TypeError('mask must have dtype bool.')
    if not (data.dtype == torch.float16 or data.dtype == torch.float32 or data.dtype == torch.float64 or (data.dtype == torch.bool) or (data.dtype == torch.int8) or (data.dtype == torch.int16) or (data.dtype == torch.int32) or (data.dtype == torch.int64)):
        raise TypeError(f'{data.dtype} is not supported in MaskedTensor.')
    if data.dim() != mask.dim():
        raise ValueError('data.dim() must equal mask.dim()')
    if data.size() != mask.size():
        raise ValueError('data.size() must equal mask.size()')