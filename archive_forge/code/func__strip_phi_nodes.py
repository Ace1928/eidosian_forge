import abc
from contextlib import contextmanager
from collections import defaultdict, namedtuple
from functools import partial
from copy import copy
import warnings
from numba.core import (errors, types, typing, ir, funcdesc, rewrites,
from numba.parfors.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfors.parfor import ParforPass as _parfor_ParforPass
from numba.parfors.parfor import ParforFusionPass as _parfor_ParforFusionPass
from numba.parfors.parfor import ParforPreLoweringPass as \
from numba.parfors.parfor import Parfor
from numba.parfors.parfor_lowering import ParforLower
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (raise_on_unsupported_feature, warn_deprecated,
from numba.core import postproc
from llvmlite import binding as llvm
def _strip_phi_nodes(self, func_ir):
    """Strip Phi nodes from ``func_ir``

        For each phi node, put incoming value to their respective incoming
        basic-block at possibly the latest position (i.e. after the latest
        assignment to the corresponding variable).
        """
    exporters = defaultdict(list)
    phis = set()
    for label, block in func_ir.blocks.items():
        for assign in block.find_insts(ir.Assign):
            if isinstance(assign.value, ir.Expr):
                if assign.value.op == 'phi':
                    phis.add(assign)
                    phi = assign.value
                    for ib, iv in zip(phi.incoming_blocks, phi.incoming_values):
                        exporters[ib].append((assign.target, iv))
    newblocks = {}
    for label, block in func_ir.blocks.items():
        newblk = copy(block)
        newblocks[label] = newblk
        newblk.body = [stmt for stmt in block.body if stmt not in phis]
        for target, rhs in exporters[label]:
            if rhs is ir.UNDEFINED:
                rhs = ir.Expr.null(loc=func_ir.loc)
            assign = ir.Assign(target=target, value=rhs, loc=rhs.loc)
            assignments = [stmt for stmt in newblk.find_insts(ir.Assign) if stmt.target == rhs]
            if assignments:
                last_assignment = assignments[-1]
                newblk.insert_after(assign, last_assignment)
            else:
                newblk.prepend(assign)
    func_ir.blocks = newblocks
    return func_ir