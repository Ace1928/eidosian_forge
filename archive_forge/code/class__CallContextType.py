from numba.core import types, errors, ir, sigutils, ir_utils
from numba.core.typing.typeof import typeof_impl
from numba.core.transforms import find_region_inout_vars
from numba.core.ir_utils import build_definitions
import numba
class _CallContextType(WithContext):
    """A simple context-manager that tells the compiler to lift the body of the
    with-block as another function.
    """

    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end, body_blocks, dispatcher_factory, extra):
        assert extra is None
        vlt = func_ir.variable_lifetime
        inputs, outputs = find_region_inout_vars(blocks=blocks, livemap=vlt.livemap, callfrom=blk_start, returnto=blk_end, body_block_ids=set(body_blocks))
        lifted_blks = {k: blocks[k] for k in body_blocks}
        _mutate_with_block_callee(lifted_blks, blk_start, blk_end, inputs, outputs)
        lifted_ir = func_ir.derive(blocks=lifted_blks, arg_names=tuple(inputs), arg_count=len(inputs), force_non_generator=True)
        dispatcher = dispatcher_factory(lifted_ir)
        newblk = _mutate_with_block_caller(dispatcher, blocks, blk_start, blk_end, inputs, outputs)
        blocks[blk_start] = newblk
        _clear_blocks(blocks, body_blocks)
        return dispatcher