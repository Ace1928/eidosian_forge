import itertools
import logging
import operator
import os
import time
from math import isclose
from pyomo.common.fileutils import find_library
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat, AbstractProblemWriter, WriterFactory
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.base import (
import pyomo.core.base.suffix
from pyomo.repn.standard_repn import generate_standard_repn
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.expression import IIdentityExpression
from pyomo.core.kernel.variable import IVariable
def _print_quad_term(self, v1, v2):
    OUTPUT = self._OUTPUT
    if v1 is not v2:
        prod_str = self._op_string[EXPR.ProductExpression]
        OUTPUT.write(prod_str)
        self._print_nonlinear_terms_NL(v1)
        self._print_nonlinear_terms_NL(v2)
    else:
        intr_expr_str = self._op_string['pow']
        OUTPUT.write(intr_expr_str)
        self._print_nonlinear_terms_NL(v1)
        OUTPUT.write(self._op_string[NumericConstant] % 2)