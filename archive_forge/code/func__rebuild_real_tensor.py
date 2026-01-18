import copy
import io
from typing import List, Union
import torch
def _rebuild_real_tensor(storage, storage_offset, size, stride):
    t = torch.tensor([], dtype=storage.dtype, device=storage._untyped_storage.device)
    return t.set_(storage._untyped_storage, storage_offset, size, stride)