from numba.core import types, errors, ir, sigutils, ir_utils
from numba.core.typing.typeof import typeof_impl
from numba.core.transforms import find_region_inout_vars
from numba.core.ir_utils import build_definitions
import numba
def _mutate_with_block_callee(blocks, blk_start, blk_end, inputs, outputs):
    """Mutate *blocks* for the callee of a with-context.

    Parameters
    ----------
    blocks : dict[ir.Block]
    blk_start, blk_end : int
        labels of the starting and ending block of the context-manager.
    inputs: sequence[str]
        Input variable names
    outputs: sequence[str]
        Output variable names
    """
    if not blocks:
        raise errors.NumbaValueError('No blocks in with-context block')
    head_blk = min(blocks)
    temp_blk = blocks[head_blk]
    scope = temp_blk.scope
    loc = temp_blk.loc
    blocks[blk_start] = ir_utils.fill_callee_prologue(block=ir.Block(scope=scope, loc=loc), inputs=inputs, label_next=head_blk)
    blocks[blk_end] = ir_utils.fill_callee_epilogue(block=ir.Block(scope=scope, loc=loc), outputs=outputs)