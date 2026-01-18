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
class AutoLinkedBooleanVar(ScalarBooleanVar):
    """A Boolean variable implicitly linked to its equivalent binary variable.

    This class provides a deprecation path for GDP.  Originally,
    Disjunct indicator_var was a binary variable.  This simplified early
    transformations.  However, with the introduction of a proper logical
    expression system, the mathematically correct approach is for the
    Disjunct's indicator_var attribute to be a proper BooleanVar.  As
    part of the transition, indicator_var attributes are instances of
    AutoLinkedBooleanVar, which allow the indicator_var to be used in
    logical expressions, but also implicitly converted (with deprecation
    warning) into their equivalent binary variable.

    Basic operations like setting values and fixing/unfixing this
    variable are also automatically applied to the associated binary
    variable.

    As this class is only intended to provide a deprecation path for
    Disjunct.indicator_var, it only supports Scalar instances and does
    not support indexing.

    """

    def as_numeric(self):
        """Return the binary variable associated with this Boolean variable.

        This method returns the associated binary variable along with a
        deprecation warning about using the Boolean variable in a numeric
        context.

        """
        deprecation_warning("Implicit conversion of the Boolean indicator_var '%s' to a binary variable is deprecated and will be removed.  Either express constraints on indicator_var using LogicalConstraints or work with the associated binary variable from indicator_var.get_associated_binary()" % (self.name,), version='6.0')
        return self.get_associated_binary()

    def as_binary(self):
        return self.as_numeric()

    def set_value(self, val, skip_validation=False, _propagate_value=True):
        super().set_value(val, skip_validation)
        if not _propagate_value:
            return
        val = self.value
        if val is not None:
            val = int(val)
        self.get_associated_binary().set_value(val, skip_validation, _propagate_value=False)

    def fix(self, value=NOTSET, skip_validation=False):
        super().fix(value, skip_validation)
        bin_var = self.get_associated_binary()
        if not bin_var.is_fixed():
            bin_var.fix()

    def unfix(self):
        super().unfix()
        bin_var = self.get_associated_binary()
        if bin_var.is_fixed():
            bin_var.unfix()

    @property
    def bounds(self):
        return self.as_numeric().bounds

    @bounds.setter
    def bounds(self, value):
        self.as_numeric().bounds = value

    @property
    def lb(self):
        return self.as_numeric().lb

    @lb.setter
    def lb(self, value):
        self.as_numeric().lb = value

    @property
    def ub(self):
        return self.as_numeric().ub

    @ub.setter
    def ub(self, value):
        self.as_numeric().ub = value

    def __abs__(self):
        return self.as_numeric().__abs__()

    def __float__(self):
        return self.as_numeric().__float__()

    def __int__(self):
        return self.as_numeric().__int__()

    def __neg__(self):
        return self.as_numeric().__neg__()

    def __bool__(self):
        return self.as_numeric().__bool__()

    def __pos__(self):
        return self.as_numeric().__pos__()

    def get_units(self):
        return self.as_numeric().get_units()

    def has_lb(self):
        return self.as_numeric().has_lb()

    def has_ub(self):
        return self.as_numeric().has_ub()

    def is_binary(self):
        return self.as_numeric().is_binary()

    def is_continuous(self):
        return self.as_numeric().is_continuous()

    def is_integer(self):
        return self.as_numeric().is_integer()

    def polynomial_degree(self):
        return self.as_numeric().polynomial_degree()

    def __le__(self, arg):
        return self.as_numeric().__le__(arg)

    def __lt__(self, arg):
        return self.as_numeric().__lt__(arg)

    def __ge__(self, arg):
        return self.as_numeric().__ge__(arg)

    def __gt__(self, arg):
        return self.as_numeric().__gt__(arg)

    def __eq__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return super().__eq__(arg)
        return self.as_numeric().__eq__(arg)

    def __ne__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return super().__ne__(arg)
        return self.as_numeric().__ne__(arg)

    def __add__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__add__(arg)

    def __div__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__div__(arg)

    def __mul__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__mul__(arg)

    def __pow__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__pow__(arg)

    def __sub__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__sub__(arg)

    def __truediv__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__truediv__(arg)

    def __iadd__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__iadd__(arg)

    def __idiv__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__idiv__(arg)

    def __imul__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__imul__(arg)

    def __ipow__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__ipow__(arg)

    def __isub__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__isub__(arg)

    def __itruediv__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__itruediv__(arg)

    def __radd__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__radd__(arg)

    def __rdiv__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__rdiv__(arg)

    def __rmul__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__rmul__(arg)

    def __rpow__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__rpow__(arg)

    def __rsub__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__rsub__(arg)

    def __rtruediv__(self, arg):
        if isinstance(arg, BooleanValue) or arg.__class__ in native_logical_types:
            return NotImplemented
        return self.as_numeric().__rtruediv__(arg)

    def setlb(self, arg):
        return self.as_numeric().setlb(arg)

    def setub(self, arg):
        return self.as_numeric().setub(arg)