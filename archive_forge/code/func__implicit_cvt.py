import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def _implicit_cvt(arg):
    if isinstance(arg, int):
        ty = str_to_ty(triton.runtime.jit.JITFunction._type_of(triton.runtime.jit.JITFunction._key_of(arg)))
        handle = TensorHandle(np.array([arg], dtype=np.int32), ty)
        return tl.tensor(handle, ty)
    if hasattr(arg, 'data_ptr'):
        ty = str_to_ty(triton.runtime.jit.JITFunction._type_of(triton.runtime.jit.JITFunction._key_of(arg)))
        handle = TensorHandle(np.array([arg.data_ptr()], dtype=np.uint64), ty)
        return tl.tensor(handle, ty)
    return arg