from numba.core import types, errors, ir, sigutils, ir_utils
from numba.core.typing.typeof import typeof_impl
from numba.core.transforms import find_region_inout_vars
from numba.core.ir_utils import build_definitions
import numba
def _bypass_with_context(blocks, blk_start, blk_end, forwardvars):
    """Given the starting and ending block of the with-context,
    replaces the head block with a new block that jumps to the end.

    *blocks* is modified inplace.
    """
    sblk = blocks[blk_start]
    scope = sblk.scope
    loc = sblk.loc
    newblk = ir.Block(scope=scope, loc=loc)
    for k, v in forwardvars.items():
        newblk.append(ir.Assign(value=scope.get_exact(k), target=scope.get_exact(v), loc=loc))
    newblk.append(ir.Jump(target=blk_end, loc=loc))
    blocks[blk_start] = newblk