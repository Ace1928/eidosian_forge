from numba.core import types, errors, ir, sigutils, ir_utils
from numba.core.typing.typeof import typeof_impl
from numba.core.transforms import find_region_inout_vars
from numba.core.ir_utils import build_definitions
import numba
def _mutate_with_block_caller(dispatcher, blocks, blk_start, blk_end, inputs, outputs):
    """Make a new block that calls into the lifeted with-context.

    Parameters
    ----------
    dispatcher : Dispatcher
    blocks : dict[ir.Block]
    blk_start, blk_end : int
        labels of the starting and ending block of the context-manager.
    inputs: sequence[str]
        Input variable names
    outputs: sequence[str]
        Output variable names
    """
    sblk = blocks[blk_start]
    scope = sblk.scope
    loc = sblk.loc
    newblock = ir.Block(scope=scope, loc=loc)
    ir_utils.fill_block_with_call(newblock=newblock, callee=dispatcher, label_next=blk_end, inputs=inputs, outputs=outputs)
    return newblock