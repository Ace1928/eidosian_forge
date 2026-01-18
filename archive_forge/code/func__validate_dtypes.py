from typing import List
import torch
def _validate_dtypes(*dtypes):
    for dtype in dtypes:
        assert isinstance(dtype, torch.dtype)
    return dtypes