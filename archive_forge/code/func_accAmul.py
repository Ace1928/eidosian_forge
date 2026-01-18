import numpy as np
from cvxpy.lin_ops.tree_mat import mul, sum_dicts, tmul
def accAmul(x, y, is_abs: bool=False):
    rows = y.shape[0]
    var_dict = vec_to_dict(x, sym_data.var_offsets, sym_data.var_sizes)
    y += constr_mul(sym_data.constraints, var_dict, rows, is_abs)