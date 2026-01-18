import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
class Sparse24TensorCutlass(Sparse24Tensor):

    def _mm(self, B: torch.Tensor, *, bias: Optional[torch.Tensor]=None, prefer_col_major_output: bool=False) -> torch.Tensor:
        if isinstance(B, Sparse24Tensor):
            raise ValueError('`Sparse24Tensor @ Sparse24Tensor` is not supported by the hardware')
        if bias is not None:
            raise NotImplementedError(f"`Sparse24Tensor` with backend='{BACKEND_CUTLASS}' does not support matmul with bias. Remove the bias, or use backend='{BACKEND_CUSPARSELT}'")
        if self.ndim != 2 or B.ndim != 2:
            raise NotImplementedError(f'`{self.__class__.__name__}` matmul: Broadcasting is not implemented')
        if self.shape[1] != B.shape[0]:
            raise NotImplementedError(f'`{self.__class__.__name__}` matmul: invalid shapes     ({self.shape[0]}, {self.shape[1]}) @ ({B.shape[0]}, {B.shape[1]})')
        return Sp24Gemm.OPERATOR(self.packed, B, self.meta)[:self.shape[0]]

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func._overloadpacket not in SPARSE24_DISPATCH_CUTLASS:
            raise NotImplementedError(f"{cls.__name__} only supports a specific set of operations, can't perform requested op ({func.__name__})")
        return SPARSE24_DISPATCH_CUTLASS[func._overloadpacket](func, types, args, kwargs)