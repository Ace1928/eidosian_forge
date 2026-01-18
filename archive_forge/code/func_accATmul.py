import numpy as np
from cvxpy.lin_ops.tree_mat import mul, sum_dicts, tmul
def accATmul(x, y, is_abs: bool=False):
    terms = constr_unpack(sym_data.constraints, x)
    val_dict = constr_tmul(sym_data.constraints, terms, is_abs)
    y += dict_to_vec(val_dict, sym_data.var_offsets, sym_data.var_sizes, sym_data.x_length)