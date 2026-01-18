import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def _finalize_casadi(casadi, available):
    if available:
        casadi_intrinsic.update({'log': casadi.log, 'log10': casadi.log10, 'sin': casadi.sin, 'cos': casadi.cos, 'tan': casadi.tan, 'cosh': casadi.cosh, 'sinh': casadi.sinh, 'tanh': casadi.tanh, 'asin': casadi.asin, 'acos': casadi.acos, 'atan': casadi.atan, 'exp': casadi.exp, 'sqrt': casadi.sqrt, 'asinh': casadi.asinh, 'acosh': casadi.acosh, 'atanh': casadi.atanh, 'ceil': casadi.ceil, 'floor': casadi.floor})