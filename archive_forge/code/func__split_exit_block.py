from collections import defaultdict, namedtuple
from contextlib import contextmanager
from copy import deepcopy, copy
import warnings
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core import (errors, types, ir, bytecode, postproc, rewrites, config,
from numba.misc.special import literal_unroll
from numba.core.analysis import (dead_branch_prune, rewrite_semantic_constants,
from numba.core.ir_utils import (guard, resolve_func_from_module, simplify_CFG,
from numba.core.ssa import reconstruct_ssa
from numba.core import interpreter
def _split_exit_block(self, fir, cfg, exit_label):
    curblock = fir.blocks[exit_label]
    newlabel = exit_label + 1
    newlabel = find_max_label(fir.blocks) + 1
    fir.blocks[newlabel] = curblock
    newblock = ir.Block(scope=curblock.scope, loc=curblock.loc)
    newblock.append(ir.Jump(newlabel, loc=curblock.loc))
    fir.blocks[exit_label] = newblock
    fir.blocks = rename_labels(fir.blocks)