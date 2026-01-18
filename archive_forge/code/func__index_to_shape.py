import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _index_to_shape(self, scope, equiv_set, var, ind_var):
    """For indexing like var[index] (either write or read), see if
        the index corresponds to a range/slice shape.
        Returns a 2-tuple where the first item is either None or a ir.Var
        to be used to replace the index variable in the outer getitem or
        setitem instruction.  The second item is also a tuple returning
        the shape and prepending instructions.
        """
    typ = self.typemap[var.name]
    require(isinstance(typ, types.ArrayCompatible))
    ind_typ = self.typemap[ind_var.name]
    ind_shape = equiv_set._get_shape(ind_var)
    var_shape = equiv_set._get_shape(var)
    if isinstance(ind_typ, types.SliceType):
        seq_typs = (ind_typ,)
        seq = (ind_var,)
    else:
        require(isinstance(ind_typ, types.BaseTuple))
        seq, op = find_build_sequence(self.func_ir, ind_var)
        require(op == 'build_tuple')
        seq_typs = tuple((self.typemap[x.name] for x in seq))
    require(len(ind_shape) == len(seq_typs) == len(var_shape))
    stmts = []

    def to_shape(typ, index, dsize):
        if isinstance(typ, types.SliceType):
            return self.slice_size(index, dsize, equiv_set, scope, stmts)
        elif isinstance(typ, types.Number):
            return (None, None)
        else:
            require(False)
    shape_list = []
    index_var_list = []
    replace_index = False
    for typ, size, dsize, orig_ind in zip(seq_typs, ind_shape, var_shape, seq):
        shape_part, index_var_part = to_shape(typ, size, dsize)
        shape_list.append(shape_part)
        if index_var_part is not None:
            replace_index = True
            index_var_list.append(index_var_part)
        else:
            index_var_list.append(orig_ind)
    if replace_index:
        if len(index_var_list) > 1:
            replacement_build_tuple_var = ir.Var(scope, mk_unique_var('replacement_build_tuple'), ind_shape[0].loc)
            new_build_tuple = ir.Expr.build_tuple(index_var_list, ind_shape[0].loc)
            stmts.append(ir.Assign(value=new_build_tuple, target=replacement_build_tuple_var, loc=ind_shape[0].loc))
            self.typemap[replacement_build_tuple_var.name] = ind_typ
        else:
            replacement_build_tuple_var = index_var_list[0]
    else:
        replacement_build_tuple_var = None
    shape = tuple(shape_list)
    require(not all((x is None for x in shape)))
    shape = tuple((x for x in shape if x is not None))
    return (replacement_build_tuple_var, ArrayAnalysis.AnalyzeResult(shape=shape, pre=stmts))