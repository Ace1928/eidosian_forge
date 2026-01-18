import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _get_or_set_rel(self, name, func_ir=None):
    """Retrieve a definition pair for the given variable,
        and if it is not already available, try to look it up
        in the given func_ir, and remember it for future use.
        """
    if isinstance(name, ir.Var):
        name = name.name
    require(self.defs.get(name, 0) == 1)
    if name in self.def_by:
        return self.def_by[name]
    else:
        require(func_ir is not None)

        def plus(x, y):
            x_is_const = isinstance(x, int)
            y_is_const = isinstance(y, int)
            if x_is_const:
                if y_is_const:
                    return x + y
                else:
                    var, offset = y
                    return (var, x + offset)
            else:
                var, offset = x
                if y_is_const:
                    return (var, y + offset)
                else:
                    return None

        def minus(x, y):
            if isinstance(y, int):
                return plus(x, -y)
            elif isinstance(x, tuple) and isinstance(y, tuple) and (x[0] == y[0]):
                return minus(x[1], y[1])
            else:
                return None
        expr = get_definition(func_ir, name)
        value = (name, 0)
        if isinstance(expr, ir.Expr):
            if expr.op == 'call':
                fname, mod_name = find_callname(func_ir, expr, typemap=self.typemap)
                if fname == 'wrap_index' and mod_name == 'numba.parfors.array_analysis':
                    index = tuple((self.obj_to_ind.get(x.name, -1) for x in expr.args))
                    if -1 in index:
                        return None
                    names = self.ext_shapes.get(index, [])
                    names.append(name)
                    if len(names) > 0:
                        self._insert(names)
                    self.ext_shapes[index] = names
            elif expr.op == 'binop':
                lhs = self._get_or_set_rel(expr.lhs, func_ir)
                rhs = self._get_or_set_rel(expr.rhs, func_ir)
                if lhs is None or rhs is None:
                    return None
                elif expr.fn == operator.add:
                    value = plus(lhs, rhs)
                elif expr.fn == operator.sub:
                    value = minus(lhs, rhs)
        elif isinstance(expr, ir.Const) and isinstance(expr.value, int):
            value = expr.value
        require(value is not None)
        self.def_by[name] = value
        if isinstance(value, int) or (isinstance(value, tuple) and (value[0] != name or value[1] != 0)):
            if isinstance(value, tuple):
                var, offset = value
                if not var in self.ref_by:
                    self.ref_by[var] = []
                self.ref_by[var].append((name, -offset))
                ind = self._get_ind(var)
                if ind >= 0:
                    objs = self.ind_to_obj[ind]
                    names = []
                    for obj in objs:
                        if obj in self.ref_by:
                            names += [x for x, i in self.ref_by[obj] if i == -offset]
                    if len(names) > 1:
                        super(SymbolicEquivSet, self)._insert(names)
        return value