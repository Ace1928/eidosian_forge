import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
class TensorAsKey:
    """A light-weight wrapper of a tensor that enables storing tensors as
    keys with efficient memory reference based comparision as an
    approximation to data equality based keys.

    Motivation: the hash value of a torch tensor is tensor instance
    based that does not use data equality and makes the usage of
    tensors as keys less useful. For instance, the result of
    ``len({a.crow_indices(), a.crow_indices()})`` is `2`, although,
    the tensor results from `crow_indices` method call are equal, in
    fact, these share the same data storage.
    On the other hand, for efficient caching of tensors we want to
    avoid calling torch.equal that compares tensors item-wise.

    TensorAsKey offers a compromise in that it guarantees key equality
    of tensors that references data in the same storage in the same
    manner and without accessing underlying data. However, this
    approach does not always guarantee correctness. For instance, for
    a complex tensor ``x``, we have ``TensorAsKey(x) ==
    TensorAsKey(x.conj())`` while ``torch.equal(x, x.conj())`` would
    return False.
    """

    def __init__(self, obj):

        def get_tensor_key(obj):
            assert not (obj.dtype.is_floating_point or obj.dtype.is_complex), obj.dtype
            return (obj.data_ptr(), obj.storage_offset(), obj.shape, obj.stride(), obj.dtype)
        self._obj_ref = weakref.ref(obj)
        if obj.layout is torch.strided:
            self.key = get_tensor_key(obj)
        elif obj.layout in {torch.sparse_csr, torch.sparse_bsr}:
            self.key = (get_tensor_key(obj.crow_indices()), get_tensor_key(obj.col_indices()))
        elif obj.layout in {torch.sparse_csc, torch.sparse_bsc}:
            self.key = (get_tensor_key(obj.ccol_indices()), get_tensor_key(obj.row_indices()))
        else:
            raise NotImplementedError(obj.layout)
        self._hash = hash(self.key)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, TensorAsKey):
            return False
        if self.obj is None or other.obj is None:
            return self is other
        return self.key == other.key

    @property
    def obj(self):
        """Return object if alive, otherwise None."""
        return self._obj_ref()