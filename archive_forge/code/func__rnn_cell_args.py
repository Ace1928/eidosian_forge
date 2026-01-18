import torch
from torch.testing._internal.common_utils import TEST_WITH_ROCM
def _rnn_cell_args(self, n, num_chunks, is_lstm, dev, dtype):
    input = (torch.randn((n, n), device=dev, dtype=torch.float32),)
    hx = ((torch.randn((n, n), device=dev, dtype=torch.float32), torch.randn((n, n), device=dev, dtype=torch.float32)) if is_lstm else torch.randn((n, n), device=dev, dtype=torch.float32),)
    weights = (torch.randn((num_chunks * n, n), device=dev, dtype=torch.float32), torch.randn((num_chunks * n, n), device=dev, dtype=torch.float32), torch.randn(num_chunks * n, device=dev, dtype=torch.float32), torch.randn(num_chunks * n, device=dev, dtype=torch.float32))
    return input + hx + weights