from collections import namedtuple
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import ZeroConstant, native_numeric_types, as_numeric
from pyomo.core import Constraint, Var, Block, Set
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.block import _BlockData
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
import logging
class _ComplementarityData(_BlockData):

    def _canonical_expression(self, e):
        e_ = None
        if e.__class__ is EXPR.EqualityExpression:
            if e.arg(1).__class__ in native_numeric_types or e.arg(1).is_fixed():
                _e = (e.arg(1), e.arg(0))
            else:
                _e = (ZeroConstant, e.arg(0) - e.arg(1))
        elif e.__class__ is EXPR.InequalityExpression:
            if e.arg(1).__class__ in native_numeric_types or e.arg(1).is_fixed():
                _e = (None, e.arg(0), e.arg(1))
            elif e.arg(0).__class__ in native_numeric_types or e.arg(0).is_fixed():
                _e = (e.arg(0), e.arg(1), None)
            else:
                _e = (ZeroConstant, e.arg(1) - e.arg(0), None)
        elif e.__class__ is EXPR.RangedExpression:
            _e = (e.arg(0), e.arg(1), e.arg(2))
        else:
            _e = (None, e, None)
        return _e

    def to_standard_form(self):
        _e1 = self._canonical_expression(self._args[0])
        _e2 = self._canonical_expression(self._args[1])
        if len(_e1) == 2:
            self.c = Constraint(expr=_e1)
            return
        if len(_e2) == 2:
            self.c = Constraint(expr=_e2)
            return
        if (_e1[0] is None) + (_e1[2] is None) + (_e2[0] is None) + (_e2[2] is None) != 2:
            raise RuntimeError('Complementarity condition %s must have exactly two finite bounds' % self.name)
        if _e1[0] is None and _e1[2] is None:
            _e1, _e2 = (_e2, _e1)
        if _e2[0] is None and _e2[2] is None:
            self.c = Constraint(expr=(None, _e2[1], None))
            self.c._complementarity_type = 3
        elif _e2[2] is None:
            self.c = Constraint(expr=_e2[0] <= _e2[1])
            self.c._complementarity_type = 1
        elif _e2[0] is None:
            self.c = Constraint(expr=-_e2[2] <= -_e2[1])
            self.c._complementarity_type = 1
        if not _e1[0] is None and (not _e1[2] is None):
            if not (_e1[0].__class__ in native_numeric_types or _e1[0].is_constant()):
                raise RuntimeError('Cannot express a complementarity problem of the form L < v < U _|_ g(x) where L is not a constant value')
            if not (_e1[2].__class__ in native_numeric_types or _e1[2].is_constant()):
                raise RuntimeError('Cannot express a complementarity problem of the form L < v < U _|_ g(x) where U is not a constant value')
            self.v = Var(bounds=(_e1[0], _e1[2]))
            self.ve = Constraint(expr=self.v == _e1[1])
        elif _e1[2] is None:
            self.v = Var(bounds=(0, None))
            self.ve = Constraint(expr=self.v == _e1[1] - _e1[0])
        else:
            self.v = Var(bounds=(0, None))
            self.ve = Constraint(expr=self.v == _e1[2] - _e1[1])

    def set_value(self, cc):
        """
        Add a complementarity condition with a specified index.
        """
        if cc.__class__ is ComplementarityTuple:
            self._args = (cc.arg0, cc.arg1)
        elif cc.__class__ is tuple:
            if len(cc) != 2:
                raise ValueError('Invalid tuple for Complementarity %s (expected 2-tuple):\n\t%s' % (self.name, cc))
            self._args = cc
        elif cc is Complementarity.Skip:
            del self.parent_component()[self.index()]
        elif cc.__class__ is list:
            return self.set_value(tuple(cc))
        else:
            raise ValueError('Unexpected value for Complementarity %s:\n\t%s' % (self.name, cc))