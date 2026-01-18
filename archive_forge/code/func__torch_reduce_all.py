import warnings
import torch
from .core import is_masked_tensor
from .creation import as_masked_tensor, masked_tensor
def _torch_reduce_all(fn):

    def reduce_all(self):
        masked_fn = _get_masked_fn(fn)
        data = self.get_data()
        mask = self.get_mask().values() if self.is_sparse else self.get_mask()
        if fn == 'all':
            result_data = masked_fn(data, mask=mask)
        elif fn in {'argmin', 'argmax'} and self.is_sparse_coo():
            sparse_idx = masked_fn(data.values(), mask=mask).to(dtype=torch.int)
            indices = data.to_sparse_coo().indices() if not self.is_sparse_coo() else data.indices()
            idx = indices.unbind(1)[sparse_idx]
            stride = data.size().numel() / torch.tensor(data.size(), device=data.device).cumprod(0)
            result_data = torch.sum(idx * stride)
        elif self.is_sparse:
            result_data = masked_fn(masked_tensor(data.values(), mask))
        else:
            result_data = masked_fn(self, mask=mask)
        return as_masked_tensor(result_data, torch.any(mask))
    return reduce_all