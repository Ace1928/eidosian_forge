import pyomo.common.unittest as unittest
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
from io import StringIO
def _constrs_contained_within(test_case, test_constr_tuples, constraint_list):
    """Checks to see if constraints defined by test_constr_tuples are in the
    constraint list.

    Parameters
    ----------
    constraint_list : Constraint
    test_constr_tuples : list of tuple
    test_case : unittest.TestCase

    """

    def _move_const_from_body(lower, repn, upper):
        if repn.constant is not None and (not repn.constant == 0):
            if lower is not None:
                lower -= repn.constant
            if upper is not None:
                upper -= repn.constant
        return (value(lower), repn, value(upper))

    def _repns_match(repn, test_repn):
        if not len(repn.linear_vars) == len(test_repn.linear_vars):
            return False
        coef_map = ComponentMap(((var, coef) for var, coef in zip(repn.linear_vars, repn.linear_coefs)))
        for var, coef in zip(test_repn.linear_vars, test_repn.linear_coefs):
            if not coef_map.get(var, 0) == coef:
                return False
        return True
    constr_list_tuples = [_move_const_from_body(constr.lower, generate_standard_repn(constr.body), constr.upper) for constr in constraint_list.values()]
    for test_lower, test_body, test_upper in test_constr_tuples:
        test_repn = generate_standard_repn(test_body)
        test_lower, test_repn, test_upper = _move_const_from_body(test_lower, test_repn, test_upper)
        found_match = False
        for lower, repn, upper in constr_list_tuples:
            if lower == test_lower and upper == test_upper and _repns_match(repn, test_repn):
                found_match = True
                break
        test_case.assertTrue(found_match, '{} <= {} <= {} was not found in constraint list.'.format(test_lower, test_body, test_upper))