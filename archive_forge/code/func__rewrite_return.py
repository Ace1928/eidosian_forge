from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def _rewrite_return(func_ir, target_block_label):
    """Rewrite a return block inside a with statement.

    Arguments
    ---------

    func_ir: Function IR
      the CFG to transform
    target_block_label: int
      the block index/label of the block containing the POP_BLOCK statement


    This implements a CFG transformation to insert a block between two other
    blocks.

    The input situation is:

    ┌───────────────┐
    │   top         │
    │   POP_BLOCK   │
    │   bottom      │
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │               │
    │    RETURN     │
    │               │
    └───────────────┘

    If such a pattern is detected in IR, it means there is a `return` statement
    within a `with` context. The basic idea is to rewrite the CFG as follows:

    ┌───────────────┐
    │   top         │
    │   POP_BLOCK   │
    │               │
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │               │
    │     bottom    │
    │               │
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │               │
    │    RETURN     │
    │               │
    └───────────────┘

    We split the block that contains the `POP_BLOCK` statement into two blocks.
    Everything from the beginning of the block up to and including the
    `POP_BLOCK` statement is considered the 'top' and everything below is
    considered 'bottom'. Finally the jump statements are re-wired to make sure
    the CFG remains valid.

    """
    target_block = func_ir.blocks[target_block_label]
    target_block_successor_label = target_block.terminator.get_targets()[0]
    target_block_successor = func_ir.blocks[target_block_successor_label]
    max_label = ir_utils.find_max_label(func_ir.blocks)
    new_label = max_label + 1
    new_block_loc = target_block_successor.loc
    new_block_scope = ir.Scope(None, loc=new_block_loc)
    new_block = ir.Block(new_block_scope, loc=new_block_loc)
    top_body, bottom_body = ([], [])
    pop_blocks = [*target_block.find_insts(ir.PopBlock)]
    assert len(pop_blocks) == 1
    assert len([*target_block.find_insts(ir.Jump)]) == 1
    assert isinstance(target_block.body[-1], ir.Jump)
    pb_marker = pop_blocks[0]
    pb_is = target_block.body.index(pb_marker)
    top_body.extend(target_block.body[:pb_is])
    top_body.append(ir.Jump(target_block_successor_label, target_block.loc))
    bottom_body.extend(target_block.body[pb_is:-1])
    bottom_body.append(ir.Jump(new_label, target_block.loc))
    return_body = func_ir.blocks[target_block_successor_label].body
    new_block.body.extend(return_body)
    target_block_successor.body.clear()
    target_block_successor.body.extend(bottom_body)
    target_block.body.clear()
    target_block.body.extend(top_body)
    func_ir.blocks[new_label] = new_block
    func_ir._definitions = ir_utils.build_definitions(func_ir.blocks)
    return func_ir