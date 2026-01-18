import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def _gen_expression(self, terms):
    terms = list(terms)
    expr = 0.0
    for term in terms:
        if type(term) is tuple:
            prodterms = list(term)
            prodexpr = 1.0
            for x in prodterms:
                prodexpr *= x
            expr += prodexpr
        else:
            expr += term
    return expr