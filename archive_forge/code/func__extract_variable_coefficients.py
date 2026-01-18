import logging
from io import StringIO
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.base import (
from pyomo.repn import generate_standard_repn
def _extract_variable_coefficients(self, row_label, repn, column_data, quadratic_data, variable_to_column):
    if len(repn.linear_coefs) > 0:
        for vardata, coef in zip(repn.linear_vars, repn.linear_coefs):
            self._referenced_variable_ids[id(vardata)] = vardata
            column_data[variable_to_column[vardata]].append((row_label, coef))
    if len(repn.quadratic_coefs) > 0:
        quad_terms = []
        for vardata, coef in zip(repn.quadratic_vars, repn.quadratic_coefs):
            self._referenced_variable_ids[id(vardata[0])] = vardata[0]
            self._referenced_variable_ids[id(vardata[1])] = vardata[1]
            quad_terms.append((vardata, coef))
        quadratic_data.append((row_label, quad_terms))
    return repn.constant