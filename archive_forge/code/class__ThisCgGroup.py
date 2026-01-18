from cupy.cuda import runtime as _runtime
from cupyx.jit import _compile
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc as _BuiltinFunc
from cupyx.jit._internal_types import Constant as _Constant
from cupyx.jit._internal_types import Data as _Data
from cupyx.jit._internal_types import wraps_class_method as _wraps_class_method
class _ThisCgGroup(_BuiltinFunc):

    def __init__(self, group_type):
        if group_type == 'grid':
            name = 'grid group'
            typename = '_GridGroup'
        elif group_type == 'thread_block':
            name = 'thread block group'
            typename = '_ThreadBlockGroup'
        else:
            raise NotImplementedError
        self.group_type = group_type
        self.__doc__ = f'\n        Returns the current {name} (:class:`~cupyx.jit.cg.{typename}`).\n\n        .. seealso:: :class:`cupyx.jit.cg.{typename}`'
        if group_type == 'grid':
            self.__doc__ += ', :func:`numba.cuda.cg.this_grid`'

    def __call__(self):
        super().__call__()

    def call_const(self, env):
        if _runtime.is_hip:
            raise RuntimeError('cooperative group is not supported on HIP')
        if self.group_type == 'grid':
            if _runtime.runtimeGetVersion() < 11000:
                raise RuntimeError('For pre-CUDA 11, the grid group has very limited functionality (only group.sync() works), and so we disable the grid group support to prepare the transition to support CUDA 11+ only.')
            cg_type = _GridGroup()
        elif self.group_type == 'thread_block':
            cg_type = _ThreadBlockGroup()
        return _Data(f'cg::this_{self.group_type}()', cg_type)