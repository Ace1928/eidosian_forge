import collections
import logging
from operator import attrgetter
from pyomo.common.config import (
from pyomo.common.dependencies import scipy, numpy as np
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import (
from pyomo.opt import WriterFactory
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import (
from pyomo.core.base import Set, RangeSet, ExternalFunction
from pyomo.network import Port
def _csc_to_nonnegative_vars(c, A, columns):
    eliminated_vars = []
    new_columns = []
    new_c_data = []
    new_c_indices = []
    new_c_indptr = [0]
    new_A_data = []
    new_A_indices = []
    new_A_indptr = [0]
    for i, v in enumerate(columns):
        lb, ub = v.bounds
        if lb is None or lb < 0:
            name = v.name
            new_columns.append(Var(name=f'_neg_{i}', domain=v.domain, bounds=(0, None if lb is None else -lb)))
            new_columns[-1].construct()
            s, e = A.indptr[i:i + 2]
            new_A_data.append(-A.data[s:e])
            new_A_indices.append(A.indices[s:e])
            new_A_indptr.append(new_A_indptr[-1] + e - s)
            s, e = c.indptr[i:i + 2]
            new_c_data.append(-c.data[s:e])
            new_c_indices.append(c.indices[s:e])
            new_c_indptr.append(new_c_indptr[-1] + e - s)
            if ub is None or ub > 0:
                new_columns.append(Var(name=f'_pos_{i}', domain=v.domain, bounds=(0, ub)))
                new_columns[-1].construct()
                s, e = A.indptr[i:i + 2]
                new_A_data.append(A.data[s:e])
                new_A_indices.append(A.indices[s:e])
                new_A_indptr.append(new_A_indptr[-1] + e - s)
                s, e = c.indptr[i:i + 2]
                new_c_data.append(c.data[s:e])
                new_c_indices.append(c.indices[s:e])
                new_c_indptr.append(new_c_indptr[-1] + e - s)
                eliminated_vars.append((v, new_columns[-1] - new_columns[-2]))
            else:
                new_columns[-1].lb = -ub
                eliminated_vars.append((v, -new_columns[-1]))
        else:
            new_columns.append(v)
            s, e = A.indptr[i:i + 2]
            new_A_data.append(A.data[s:e])
            new_A_indices.append(A.indices[s:e])
            new_A_indptr.append(new_A_indptr[-1] + e - s)
            s, e = c.indptr[i:i + 2]
            new_c_data.append(c.data[s:e])
            new_c_indices.append(c.indices[s:e])
            new_c_indptr.append(new_c_indptr[-1] + e - s)
    nCol = len(new_columns)
    c = scipy.sparse.csc_array((np.concatenate(new_c_data), np.concatenate(new_c_indices), new_c_indptr), [c.shape[0], nCol])
    A = scipy.sparse.csc_array((np.concatenate(new_A_data), np.concatenate(new_A_indices), new_A_indptr), [A.shape[0], nCol])
    return (c, A, new_columns, eliminated_vars)