from numba.core import types
from numba.core.extending import overload, overload_method
from numba.core.typing import signature
from numba.cuda import nvvmutils
from numba.cuda.extending import intrinsic
from numba.cuda.types import grid_group, GridGroup as GridGroupClass
@overload_method(GridGroupClass, 'sync', target='cuda')
def _ol_grid_group_sync(group):

    def impl(group):
        return _grid_group_sync(group)
    return impl