import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_op_call_numpy_reshape(self, scope, equiv_set, loc, args, kws):
    n = len(args)
    assert n > 1
    if n == 2:
        typ = self.typemap[args[1].name]
        if isinstance(typ, types.BaseTuple):
            return ArrayAnalysis.AnalyzeResult(shape=args[1])
    stmts = []
    neg_one_index = -1
    for arg_index in range(1, len(args)):
        reshape_arg = args[arg_index]
        reshape_arg_def = guard(get_definition, self.func_ir, reshape_arg)
        if isinstance(reshape_arg_def, ir.Const):
            if reshape_arg_def.value < 0:
                if neg_one_index == -1:
                    neg_one_index = arg_index
                else:
                    msg = 'The reshape API may only include one negative argument.'
                    raise errors.UnsupportedRewriteError(msg, loc=reshape_arg.loc)
    if neg_one_index >= 0:
        loc = args[0].loc
        calc_size_var = ir.Var(scope, mk_unique_var('calc_size_var'), loc)
        self.typemap[calc_size_var.name] = types.intp
        init_calc_var = ir.Assign(ir.Expr.getattr(args[0], 'size', loc), calc_size_var, loc)
        stmts.append(init_calc_var)
        for arg_index in range(1, len(args)):
            if arg_index == neg_one_index:
                continue
            div_calc_size_var = ir.Var(scope, mk_unique_var('calc_size_var'), loc)
            self.typemap[div_calc_size_var.name] = types.intp
            new_binop = ir.Expr.binop(operator.floordiv, calc_size_var, args[arg_index], loc)
            div_calc = ir.Assign(new_binop, div_calc_size_var, loc)
            self.calltypes[new_binop] = signature(types.intp, types.intp, types.intp)
            stmts.append(div_calc)
            calc_size_var = div_calc_size_var
        args[neg_one_index] = calc_size_var
    return ArrayAnalysis.AnalyzeResult(shape=tuple(args[1:]), pre=stmts)