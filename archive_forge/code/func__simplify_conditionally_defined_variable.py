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
def _simplify_conditionally_defined_variable(self, func_ir):
    """
        Rewrite assignments like:

            ver1 = null()
            ...
            ver1 = ver
            ...
            uses(ver1)

        into:
            # delete all assignments to ver1
            uses(ver)

        This is only needed for parfors because the SSA pass will create extra
        variable assignments that the parfor code does not expect.
        This pass helps avoid problems by reverting the effect of SSA.
        """
    any_block = next(iter(func_ir.blocks.values()))
    scope = any_block.scope
    defs = func_ir._definitions

    def unver_or_undef(unver, defn):
        if isinstance(defn, ir.Var):
            if defn.unversioned_name == unver:
                return True
        elif isinstance(defn, ir.Expr):
            if defn.op == 'null':
                return True
        return False

    def legalize_all_versioned_names(var):
        if not var.versioned_names:
            return False
        for versioned in var.versioned_names:
            vs = defs.get(versioned, ())
            if not all(map(partial(unver_or_undef, k), vs)):
                return False
        return True
    suspects = set()
    for k in defs:
        try:
            var = scope.get_exact(k)
        except errors.NotDefinedError:
            continue
        if var.unversioned_name == k:
            if legalize_all_versioned_names(var):
                suspects.add(var)
    delete_set = set()
    replace_map = {}
    for var in suspects:
        for versioned in var.versioned_names:
            ver_var = scope.get_exact(versioned)
            delete_set.add(ver_var)
            replace_map[versioned] = var
    for _label, blk in func_ir.blocks.items():
        for assign in blk.find_insts(ir.Assign):
            if assign.target in delete_set:
                blk.remove(assign)
    replace_vars(func_ir.blocks, replace_map)