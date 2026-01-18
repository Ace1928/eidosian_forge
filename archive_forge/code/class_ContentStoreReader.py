import ctypes
import functools
import hashlib
import os.path
import struct
from collections import defaultdict
from typing import Dict, Optional, Set
import torch
import torch._prims as prims
import torch._utils
import torch.nn.functional as F
from torch._C import default_generator
from torch.multiprocessing.reductions import StorageWeakRef
class ContentStoreReader:

    def __init__(self, loc: str, *, cache=True) -> None:
        self.loc = loc
        self.storage_cache: Optional[Dict[Optional[torch.device], Dict[str, StorageWeakRef]]] = None
        if cache:
            self.storage_cache = defaultdict(dict)

    def read_storage(self, h: str, *, device=None) -> torch.UntypedStorage:
        if device is not None:
            device = torch.device(device)
        ws = self.storage_cache[device].get(h) if self.storage_cache is not None else None
        s: Optional[torch.UntypedStorage]
        if ws is not None:
            s = torch.UntypedStorage._new_with_weak_ptr(ws.cdata)
            if s is not None:
                return s
        s = torch.load(os.path.join(self.loc, 'storages', h), weights_only=True, map_location=device)._untyped_storage
        assert s is not None
        if self.storage_cache is not None:
            self.storage_cache[device][h] = StorageWeakRef(s)
        return s

    def read_tensor_metadata(self, name: str):
        fn = os.path.join(self.loc, 'tensors', name)
        if not os.path.exists(fn):
            raise FileNotFoundError(fn)
        return torch.load(fn, weights_only=True)

    def read_tensor(self, name: str, *, device=None) -> torch.Tensor:
        dtype, h, storage_offset, size, stride, metadata = self.read_tensor_metadata(name)
        storage = self.read_storage(h, device=device)
        t = torch.tensor([], dtype=dtype, device=storage.device)
        t.set_(storage, storage_offset, size, stride)
        torch._utils.set_tensor_metadata(t, metadata)
        return t