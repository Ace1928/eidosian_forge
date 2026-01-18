from typing import Any, Mapping
import warnings
import cupy
from cupy_backends.cuda.api import runtime
from cupy.cuda import device
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit._internal_types import BuiltinFunc
from cupyx.jit._internal_types import Data
from cupyx.jit._internal_types import Constant
from cupyx.jit._internal_types import Range
from cupyx.jit import _compile
from functools import reduce
class WarpShuffleOp(BuiltinFunc):

    def __init__(self, op, dtypes):
        self._op = op
        self._name = '__shfl_' + (op + '_' if op else '') + 'sync'
        self._dtypes = dtypes
        doc = f'Calls the ``{self._name}`` function. Please refer to\n        `Warp Shuffle Functions`_ for detailed explanation.\n\n        .. _Warp Shuffle Functions:\n            https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions\n        '
        self.__doc__ = doc

    def __call__(self, mask, var, val_id, *, width=32):
        super().__call__()

    def call(self, env, mask, var, val_id, *, width=None):
        name = self._name
        var = Data.init(var, env)
        ctype = var.ctype
        if ctype.dtype.name not in self._dtypes:
            raise TypeError(f'`{name}` does not support {ctype.dtype} input.')
        try:
            mask = mask.obj
        except Exception:
            raise TypeError('mask must be an integer')
        if runtime.is_hip:
            warnings.warn(f'mask {mask} is ignored on HIP', RuntimeWarning)
        elif not 0 <= mask <= 4294967295:
            raise ValueError('mask is out of range')
        if self._op in ('up', 'down'):
            val_id_t = _cuda_types.uint32
        else:
            val_id_t = _cuda_types.int32
        val_id = _compile._astype_scalar(val_id, val_id_t, 'same_kind', env)
        val_id = Data.init(val_id, env)
        if width:
            if isinstance(width, Constant):
                if width.obj not in (2, 4, 8, 16, 32):
                    raise ValueError('width needs to be power of 2')
        else:
            width = Constant(64) if runtime.is_hip else Constant(32)
        width = _compile._astype_scalar(width, _cuda_types.int32, 'same_kind', env)
        width = Data.init(width, env)
        code = f'{name}({hex(mask)}, {var.code}, {val_id.code}'
        code += f', {width.code})'
        return Data(code, ctype)