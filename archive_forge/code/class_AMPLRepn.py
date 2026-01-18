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
class AMPLRepn(object):
    __slots__ = ('nl', 'mult', 'const', 'linear', 'nonlinear', 'named_exprs')
    ActiveVisitor = None

    def __init__(self, const, linear, nonlinear):
        self.nl = None
        self.mult = 1
        self.const = const
        self.linear = linear
        if nonlinear is None:
            self.nonlinear = self.named_exprs = None
        else:
            nl, nl_args, self.named_exprs = nonlinear
            self.nonlinear = (nl, nl_args)

    def __str__(self):
        return f'AMPLRepn(mult={self.mult}, const={self.const}, linear={self.linear}, nonlinear={self.nonlinear}, nl={self.nl}, named_exprs={self.named_exprs})'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return other.__class__ is AMPLRepn and (self.nl == other.nl and self.mult == other.mult and (self.const == other.const) and (self.linear == other.linear) and (self.nonlinear == other.nonlinear) and (self.named_exprs == other.named_exprs))

    def __hash__(self):
        return id(self) // 16 + ((id(self) & 15) << 8 * ctypes.sizeof(ctypes.c_void_p) - 4)

    def duplicate(self):
        ans = self.__class__.__new__(self.__class__)
        ans.nl = self.nl
        ans.mult = self.mult
        ans.const = self.const
        ans.linear = None if self.linear is None else dict(self.linear)
        ans.nonlinear = self.nonlinear
        ans.named_exprs = self.named_exprs
        return ans

    def compile_repn(self, visitor, prefix='', args=None, named_exprs=None):
        template = visitor.template
        if self.mult != 1:
            if self.mult == -1:
                prefix += template.negation
            else:
                prefix += template.multiplier % self.mult
            self.mult = 1
        if self.named_exprs is not None:
            if named_exprs is None:
                named_exprs = set(self.named_exprs)
            else:
                named_exprs.update(self.named_exprs)
        if self.nl is not None:
            nl, nl_args = self.nl
            if prefix:
                nl = prefix + nl
            if args is not None:
                assert args is not nl_args
                args.extend(nl_args)
            else:
                args = list(nl_args)
            if nl_args:
                named_exprs.update(nl_args)
            return (nl, args, named_exprs)
        if args is None:
            args = []
        if self.linear:
            nterms = -len(args)
            _v_template = template.var
            _m_template = template.monomial
            nl_sum = ''.join((args.append(v) or (_v_template if c == 1 else _m_template % c) for v, c in self.linear.items() if c))
            nterms += len(args)
        else:
            nterms = 0
            nl_sum = ''
        if self.nonlinear:
            if self.nonlinear.__class__ is list:
                nterms += len(self.nonlinear)
                nl_sum += ''.join(map(itemgetter(0), self.nonlinear))
                deque(map(args.extend, map(itemgetter(1), self.nonlinear)), maxlen=0)
            else:
                nterms += 1
                nl_sum += self.nonlinear[0]
                args.extend(self.nonlinear[1])
        if self.const:
            nterms += 1
            nl_sum += template.const % self.const
        if nterms > 2:
            return (prefix + template.nary_sum % nterms + nl_sum, args, named_exprs)
        elif nterms == 2:
            return (prefix + template.binary_sum + nl_sum, args, named_exprs)
        elif nterms == 1:
            return (prefix + nl_sum, args, named_exprs)
        else:
            return (prefix + template.const % 0, args, named_exprs)

    def compile_nonlinear_fragment(self, visitor):
        if not self.nonlinear:
            self.nonlinear = None
            return
        args = []
        nterms = len(self.nonlinear)
        nl_sum = ''.join(map(itemgetter(0), self.nonlinear))
        deque(map(args.extend, map(itemgetter(1), self.nonlinear)), maxlen=0)
        if nterms > 2:
            self.nonlinear = (visitor.template.nary_sum % nterms + nl_sum, args)
        elif nterms == 2:
            self.nonlinear = (visitor.template.binary_sum + nl_sum, args)
        else:
            self.nonlinear = (nl_sum, args)

    def append(self, other):
        """Append a child result from acceptChildResult

        Notes
        -----
        This method assumes that the operator was "+". It is implemented
        so that we can directly use an AMPLRepn() as a data object in
        the expression walker (thereby avoiding the function call for a
        custom callback)

        """
        _type = other[0]
        if _type is _MONOMIAL:
            _, v, c = other
            if v in self.linear:
                self.linear[v] += c
            else:
                self.linear[v] = c
        elif _type is _GENERAL:
            _, other = other
            if other.nl is not None and other.nl[1]:
                if other.linear:
                    pass
                else:
                    other = other.compile_repn(self.ActiveVisitor, '', None, self.named_exprs)
                    nl, nl_args, self.named_exprs = other
                    self.nonlinear.append((nl, nl_args))
                    return
            if other.named_exprs is not None:
                if self.named_exprs is None:
                    self.named_exprs = set(other.named_exprs)
                else:
                    self.named_exprs.update(other.named_exprs)
            if other.mult != 1:
                mult = other.mult
                self.const += mult * other.const
                if other.linear:
                    linear = self.linear
                    for v, c in other.linear.items():
                        if v in linear:
                            linear[v] += c * mult
                        else:
                            linear[v] = c * mult
                if other.nonlinear:
                    if other.nonlinear.__class__ is list:
                        other.compile_nonlinear_fragment(self.ActiveVisitor)
                    if mult == -1:
                        prefix = self.ActiveVisitor.template.negation
                    else:
                        prefix = self.ActiveVisitor.template.multiplier % mult
                    self.nonlinear.append((prefix + other.nonlinear[0], other.nonlinear[1]))
            else:
                self.const += other.const
                if other.linear:
                    linear = self.linear
                    for v, c in other.linear.items():
                        if v in linear:
                            linear[v] += c
                        else:
                            linear[v] = c
                if other.nonlinear:
                    if other.nonlinear.__class__ is list:
                        self.nonlinear.extend(other.nonlinear)
                    else:
                        self.nonlinear.append(other.nonlinear)
        elif _type is _CONSTANT:
            self.const += other[1]

    def to_expr(self, var_map):
        if self.nl is not None or self.nonlinear is not None:
            raise MouseTrap('Cannot convert nonlinear AMPLRepn to Pyomo Expression')
        if self.linear:
            ans = LinearExpression([coef * var_map[vid] for vid, coef in self.linear.items()])
            ans += self.const
        else:
            ans = self.const
        return ans * self.mult