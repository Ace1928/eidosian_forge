import ctypes
import logging
import os
from collections import deque, defaultdict, namedtuple
from contextlib import nullcontext
from itertools import filterfalse, product
from math import log10 as _log10
from operator import itemgetter, attrgetter, setitem
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException, MouseTrap
from pyomo.common.gc_manager import PauseGC
from pyomo.common.numeric_types import (
from pyomo.common.timing import TicTocTimer
from pyomo.core.expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import (
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.base.var import _VarData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
from pyomo.opt import WriterFactory
from pyomo.repn.util import (
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port
class text_nl_debug_template(object):
    unary = {'log': 'o43\t#log\n', 'log10': 'o42\t#log10\n', 'sin': 'o41\t#sin\n', 'cos': 'o46\t#cos\n', 'tan': 'o38\t#tan\n', 'sinh': 'o40\t#sinh\n', 'cosh': 'o45\t#cosh\n', 'tanh': 'o37\t#tanh\n', 'asin': 'o51\t#asin\n', 'acos': 'o53\t#acos\n', 'atan': 'o49\t#atan\n', 'exp': 'o44\t#exp\n', 'sqrt': 'o39\t#sqrt\n', 'asinh': 'o50\t#asinh\n', 'acosh': 'o52\t#acosh\n', 'atanh': 'o47\t#atanh\n', 'ceil': 'o14\t#ceil\n', 'floor': 'o13\t#floor\n'}
    binary_sum = 'o0\t#+\n'
    product = 'o2\t#*\n'
    division = 'o3\t# /\n'
    pow = 'o5\t#^\n'
    abs = 'o15\t# abs\n'
    negation = 'o16\t#-\n'
    nary_sum = 'o54\t# sumlist\n%d\t# (n)\n'
    exprif = 'o35\t# if\n'
    and_expr = 'o21\t# and\n'
    less_than = 'o22\t# lt\n'
    less_equal = 'o23\t# le\n'
    equality = 'o24\t# eq\n'
    external_fcn = 'f%d %d%s\n'
    var = '%s\n'
    const = 'n%r\n'
    string = 'h%d:%s\n'
    monomial = product + const + var.replace('%', '%%')
    multiplier = product + const
    _create_strict_inequality_map(vars())