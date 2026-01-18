import ctypes
import logging
import os
from pyomo.common.fileutils import Library
from pyomo.core import value, Expression
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, identify_variables
from pyomo.common.collections import ComponentMap
def _MCPP_lib():
    """A singleton interface to the MC++ library"""
    if _MCPP_lib._mcpp is not None:
        return _MCPP_lib._mcpp
    _MCPP_lib._mcpp = mcpp = ctypes.CDLL(Library('mcppInterface').path())
    mcpp.get_version.restype = ctypes.c_char_p
    mcpp.toString.argtypes = [ctypes.c_void_p]
    mcpp.toString.restype = ctypes.c_char_p
    mcpp.lower.argtypes = [ctypes.c_void_p]
    mcpp.lower.restype = ctypes.c_double
    mcpp.upper.argtypes = [ctypes.c_void_p]
    mcpp.upper.restype = ctypes.c_double
    mcpp.concave.argtypes = [ctypes.c_void_p]
    mcpp.concave.restype = ctypes.c_double
    mcpp.convex.argtypes = [ctypes.c_void_p]
    mcpp.convex.restype = ctypes.c_double
    mcpp.subcc.argtypes = [ctypes.c_void_p, ctypes.c_int]
    mcpp.subcc.restype = ctypes.c_double
    mcpp.subcv.argtypes = [ctypes.c_void_p, ctypes.c_int]
    mcpp.subcv.restype = ctypes.c_double
    mcpp.newVar.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
    mcpp.newVar.restype = ctypes.c_void_p
    mcpp.newConstant.argtypes = [ctypes.c_double]
    mcpp.newConstant.restype = ctypes.c_void_p
    mcpp.multiply.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcpp.multiply.restype = ctypes.c_void_p
    mcpp.add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcpp.add.restype = ctypes.c_void_p
    mcpp.power.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcpp.power.restype = ctypes.c_void_p
    mcpp.powerf.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcpp.powerf.restype = ctypes.c_void_p
    mcpp.powerx.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcpp.powerx.restype = ctypes.c_void_p
    mcpp.mc_sqrt.argtypes = [ctypes.c_void_p]
    mcpp.mc_sqrt.restype = ctypes.c_void_p
    mcpp.negation.argtypes = [ctypes.c_void_p]
    mcpp.negation.restype = ctypes.c_void_p
    mcpp.mc_abs.argtypes = [ctypes.c_void_p]
    mcpp.mc_abs.restype = ctypes.c_void_p
    mcpp.trigSin.argtypes = [ctypes.c_void_p]
    mcpp.trigSin.restype = ctypes.c_void_p
    mcpp.trigCos.argtypes = [ctypes.c_void_p]
    mcpp.trigCos.restype = ctypes.c_void_p
    mcpp.trigTan.argtypes = [ctypes.c_void_p]
    mcpp.trigTan.restype = ctypes.c_void_p
    mcpp.atrigSin.argtypes = [ctypes.c_void_p]
    mcpp.atrigSin.restype = ctypes.c_void_p
    mcpp.atrigCos.argtypes = [ctypes.c_void_p]
    mcpp.atrigCos.restype = ctypes.c_void_p
    mcpp.atrigTan.argtypes = [ctypes.c_void_p]
    mcpp.atrigTan.restype = ctypes.c_void_p
    mcpp.exponential.argtypes = [ctypes.c_void_p]
    mcpp.exponential.restype = ctypes.c_void_p
    mcpp.logarithm.argtypes = [ctypes.c_void_p]
    mcpp.logarithm.restype = ctypes.c_void_p
    mcpp.release.argtypes = [ctypes.c_void_p]
    mcpp.try_unary_fcn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcpp.try_unary_fcn.restype = ctypes.c_void_p
    mcpp.try_binary_fcn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    mcpp.try_binary_fcn.restype = ctypes.c_void_p
    mcpp.get_last_exception_message.restype = ctypes.c_char_p
    return mcpp