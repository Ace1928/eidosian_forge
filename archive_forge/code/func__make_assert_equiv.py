import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _make_assert_equiv(self, scope, loc, equiv_set, _args, names=None):
    if config.DEBUG_ARRAY_OPT >= 2:
        print('make_assert_equiv:', _args, names)
    if names is None:
        names = [x.name for x in _args]
    args = []
    arg_names = []
    for name, x in zip(names, _args):
        if config.DEBUG_ARRAY_OPT >= 2:
            print('name, x:', name, x)
        seen = False
        for y in args:
            if config.DEBUG_ARRAY_OPT >= 2:
                print('is equiv to?', y, equiv_set.is_equiv(x, y))
            if equiv_set.is_equiv(x, y):
                seen = True
                break
        if not seen:
            args.append(x)
            arg_names.append(name)
    if len(args) < 2:
        if config.DEBUG_ARRAY_OPT >= 2:
            print('Will not insert assert_equiv as args are known to be equivalent.')
        return []
    msg = 'Sizes of {} do not match on {}'.format(', '.join(arg_names), loc)
    msg_val = ir.Const(msg, loc)
    msg_typ = types.StringLiteral(msg)
    msg_var = ir.Var(scope, mk_unique_var('msg'), loc)
    self.typemap[msg_var.name] = msg_typ
    argtyps = tuple([msg_typ] + [self.typemap[x.name] for x in args])
    tup_typ = types.StarArgTuple.from_types(argtyps)
    assert_var = ir.Var(scope, mk_unique_var('assert'), loc)
    assert_def = ir.Global('assert_equiv', assert_equiv, loc=loc)
    fnty = get_global_func_typ(assert_equiv)
    sig = self.context.resolve_function_type(fnty, (tup_typ,), {})
    self._define(equiv_set, assert_var, fnty, assert_def)
    var = ir.Var(scope, mk_unique_var('ret'), loc)
    value = ir.Expr.call(assert_var, [msg_var] + args, {}, loc=loc)
    self._define(equiv_set, var, types.none, value)
    self.calltypes[value] = sig
    return [ir.Assign(value=msg_val, target=msg_var, loc=loc), ir.Assign(value=assert_def, target=assert_var, loc=loc), ir.Assign(value=value, target=var, loc=loc)]