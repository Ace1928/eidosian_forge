from cupy.cuda import runtime as _runtime
from cupyx.jit import _compile
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc as _BuiltinFunc
from cupyx.jit._internal_types import Constant as _Constant
from cupyx.jit._internal_types import Data as _Data
from cupyx.jit._internal_types import wraps_class_method as _wraps_class_method
class _Sync(_BuiltinFunc):

    def __call__(self, group):
        """Calls ``cg::sync()``.

        Args:
            group: a valid cooperative group

        .. seealso:: `cg::sync`_

        .. _cg::sync:
            https://docs.nvidia.com/cuda/archive/11.6.0/cuda-c-programming-guide/index.html#collectives-cg-sync
        """
        super().__call__()

    def call(self, env, group):
        if _runtime.runtimeGetVersion() < 11000:
            raise RuntimeError('not supported in CUDA < 11.0')
        if not isinstance(group.ctype, _ThreadGroup):
            raise ValueError('group must be a valid cooperative group')
        _check_include(env, 'cg')
        return _Data(f'cg::sync({group.code})', _cuda_types.void)