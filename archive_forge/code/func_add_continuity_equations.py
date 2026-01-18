import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core import Suffix, Var, Constraint, Piecewise, Block
from pyomo.core import Expression, Param
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.block import IndexedBlock, SortComponents
from pyomo.dae import ContinuousSet, DAE_Error
from pyomo.common.formatting import tostr
from io import StringIO
def add_continuity_equations(block, d, i, loc):
    """
    Adds continuity equations in the case that the polynomial basis function
    does not have a root at the finite element boundary
    """
    svar = d.get_state_var()
    nme = svar.local_name + '_' + i.local_name + '_cont_eq'
    if block.find_component(nme) is not None:
        return

    def _cont_exp(v, s):
        ncp = s.get_discretization_info()['ncp']
        afinal = s.get_discretization_info()['afinal']

        def _fun(i):
            tmp = list(s)
            idx = s.ord(i) - 1
            low = s.get_lower_element_boundary(i)
            if i != low or idx == 0:
                raise IndexError('list index out of range')
            low = s.get_lower_element_boundary(tmp[idx - 1])
            lowidx = s.ord(low) - 1
            return sum((v(tmp[lowidx + j]) * afinal[j] for j in range(ncp + 1)))
        return _fun
    expr = create_partial_expression(_cont_exp, create_access_function(svar), i, loc)

    def _cont_eq(m, *args):
        try:
            return svar[args] == expr(*args)
        except IndexError:
            return Constraint.Skip
    block.add_component(nme, Constraint(d.index_set(), rule=_cont_eq))