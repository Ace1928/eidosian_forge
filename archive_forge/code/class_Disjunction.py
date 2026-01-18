import logging
import sys
import types
from math import fabs
from weakref import ref as weakref_ref
from pyomo.common.autoslots import AutoSlots
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.errors import PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import native_logical_types, native_types
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core import (
from pyomo.core.base.component import (
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.block import _BlockData
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.indexed_component import ActiveIndexedComponent
from pyomo.core.expr.expr_common import ExpressionType
@ModelComponentFactory.register('Disjunction expressions.')
class Disjunction(ActiveIndexedComponent):
    _ComponentDataClass = _DisjunctionData

    def __new__(cls, *args, **kwds):
        if cls != Disjunction:
            return super(Disjunction, cls).__new__(cls)
        if args == ():
            return ScalarDisjunction.__new__(ScalarDisjunction)
        else:
            return IndexedDisjunction.__new__(IndexedDisjunction)

    def __init__(self, *args, **kwargs):
        self._init_rule = kwargs.pop('rule', None)
        self._init_expr = kwargs.pop('expr', None)
        self._init_xor = _Initializer.process(kwargs.pop('xor', True))
        self._autodisjuncts = None
        kwargs.setdefault('ctype', Disjunction)
        super(Disjunction, self).__init__(*args, **kwargs)
        if self._init_expr is not None and self._init_rule is not None:
            raise ValueError('Cannot specify both rule= and expr= for Disjunction %s' % (self.name,))

    def _setitem_impl(self, index, obj, value):
        if value is Disjunction.Skip:
            del self[index]
            return None
        else:
            obj.set_value(value)
            return obj

    def _setitem_when_not_present(self, index, value):
        if value is Disjunction.Skip:
            return None
        else:
            ans = super(Disjunction, self)._setitem_when_not_present(index=index, value=value)
            self._initialize_members((index,))
            return ans

    def _initialize_members(self, init_set):
        if self._init_xor[0] == _Initializer.value:
            val = self._init_xor[1]
            for key in init_set:
                self._data[key].xor = val
        elif self._init_xor[0] == _Initializer.deferred_value:
            val = bool(value(self._init_xor[1]))
            for key in init_set:
                self._data[key].xor = val
        elif self._init_xor[0] == _Initializer.function:
            fcn = self._init_xor[1]
            for key in init_set:
                self._data[key].xor = bool(value(apply_indexed_rule(self, fcn, self._parent(), key)))
        elif self._init_xor[0] == _Initializer.dict_like:
            val = self._init_xor[1]
            for key in init_set:
                self._data[key].xor = bool(value(val[key]))

    def construct(self, data=None):
        if is_debug_set(logger):
            logger.debug('Constructing disjunction %s' % self.name)
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed = True
        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()
        _self_parent = self.parent_block()
        if not self.is_indexed():
            if self._init_rule is not None:
                expr = self._init_rule(_self_parent)
            elif self._init_expr is not None:
                expr = self._init_expr
            else:
                timer.report()
                return
            if expr is None:
                raise ValueError(_rule_returned_none_error % (self.name,))
            if expr is Disjunction.Skip:
                timer.report()
                return
            self._data[None] = self
            self._setitem_when_not_present(None, expr)
        elif self._init_expr is not None:
            raise IndexError("Disjunction '%s': Cannot initialize multiple indices of a disjunction with a single disjunction list" % (self.name,))
        elif self._init_rule is not None:
            _init_rule = self._init_rule
            for ndx in self._index_set:
                try:
                    expr = apply_indexed_rule(self, _init_rule, _self_parent, ndx)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error('Rule failed when generating expression for disjunction %s with index %s:\n%s: %s' % (self.name, str(ndx), type(err).__name__, err))
                    raise
                if expr is None:
                    _name = '%s[%s]' % (self.name, str(ndx))
                    raise ValueError(_rule_returned_none_error % (_name,))
                if expr is Disjunction.Skip:
                    continue
                self._setitem_when_not_present(ndx, expr)
        timer.report()

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return ([('Size', len(self)), ('Index', self._index_set if self.is_indexed() else None), ('Active', self.active)], self.items(), ('Disjuncts', 'Active', 'XOR'), lambda k, v: [[x.name for x in v.disjuncts], v.active, v.xor])