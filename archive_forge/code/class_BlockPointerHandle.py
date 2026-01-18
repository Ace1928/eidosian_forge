import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
class BlockPointerHandle:

    def __init__(self, base, shape, strides, offsets, tensor_shape, order):
        self.base = base
        self.shape = shape
        self.strides = strides
        self.offsets = offsets
        self.tensor_shape = tensor_shape
        self.order = order

    def materialize_pointers(self, boundary_check):
        dtype_tt = self.base.dtype.element_ty
        n_bytes = dtype_tt.primitive_bitwidth // 8
        tensor_shape = self.tensor_shape
        ptrs = np.broadcast_to(self.base.data, self.tensor_shape)
        masks = np.ones(self.tensor_shape, dtype=bool)
        for dim in range(len(tensor_shape)):
            bcast_dims = [1] * len(tensor_shape)
            bcast_dims[dim] = tensor_shape[dim]
            off = (self.offsets[dim].data + np.arange(tensor_shape[dim])).reshape(bcast_dims)
            ptrs = ptrs + (n_bytes * off * self.strides[dim].data).astype(np.uint64)
            if dim in boundary_check:
                masks = np.logical_and(masks, off < self.shape[dim].data)
        ptrs = TensorHandle(ptrs, self.base.dtype)
        return (ptrs, masks)