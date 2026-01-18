import inspect
import logging
import sys
import textwrap
import pyomo.core.expr as EXPR
import pyomo.core.base as BASE
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.component import Component, ActiveComponent
from pyomo.core.base.config import PyomoOptions
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_set
from pyomo.core.expr.numeric_expr import _ndarray
from pyomo.core.pyomoobject import PyomoObject
from pyomo.common import DeveloperError
from pyomo.common.autoslots import fast_deepcopy
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types
from pyomo.common.sorting import sorted_robust
from collections.abc import Sequence
def _processUnhashableIndex(self, idx):
    """Process a call to __getitem__ with unhashable elements

        There are three basic ways to get here:
          1) the index contains one or more slices or ellipsis
          2) the index contains an unhashable type (e.g., a Pyomo
             (Scalar)Component)
          3) the index contains an IndexTemplate
        """
    orig_idx = idx
    fixed = {}
    sliced = {}
    ellipsis = None
    if normalize_index.flatten:
        idx = normalize_index(idx)
    if idx.__class__ is not tuple:
        idx = (idx,)
    for i, val in enumerate(idx):
        if type(val) is slice:
            if val.start is not None or val.stop is not None or val.step is not None:
                raise IndexError('Indexed components can only be indexed with simple slices: start and stop values are not allowed.')
            else:
                if ellipsis is None:
                    sliced[i] = val
                else:
                    sliced[i - len(idx)] = val
                continue
        if val is Ellipsis:
            if ellipsis is not None:
                raise IndexError("Indexed components can only be indexed with simple slices: the Pyomo wildcard slice (Ellipsis; e.g., '...') can only appear once")
            ellipsis = i
            continue
        if hasattr(val, 'is_expression_type'):
            _num_val = val
            try:
                val = EXPR.evaluate_expression(val, constant=True)
            except TemplateExpressionError:
                return EXPR.GetItemExpression((self,) + tuple(idx))
            except EXPR.NonConstantExpressionError:
                raise RuntimeError('Error retrieving the value of an indexed item %s:\nindex %s is not a constant value.  This is likely not what you meant to\ndo, as if you later change the fixed value of the object this lookup\nwill not change.  If you understand the implications of using\nnon-constant values, you can get the current value of the object using\nthe value() function.' % (self.name, i))
            except EXPR.FixedExpressionError:
                raise RuntimeError('Error retrieving the value of an indexed item %s:\nindex %s is a fixed but not constant value.  This is likely not what you\nmeant to do, as if you later change the fixed value of the object this\nlookup will not change.  If you understand the implications of using\nfixed but not constant values, you can get the current value using the\nvalue() function.' % (self.name, i))
        hash(val)
        if ellipsis is None:
            fixed[i] = val
        else:
            fixed[i - len(idx)] = val
    if sliced or ellipsis is not None:
        slice_dim = len(idx)
        if ellipsis is not None:
            slice_dim -= 1
        if normalize_index.flatten:
            set_dim = self.dim()
        elif not self.is_indexed():
            set_dim = 0
        else:
            set_dim = self.index_set().dimen
            if set_dim is None:
                set_dim = 1
        structurally_valid = False
        if slice_dim == set_dim or set_dim is None:
            structurally_valid = True
        elif type(set_dim) is type:
            pass
        elif ellipsis is not None and slice_dim < set_dim:
            structurally_valid = True
        elif set_dim == 0 and idx == (slice(None),):
            structurally_valid = True
        if not structurally_valid:
            msg = "Index %s contains an invalid number of entries for component '%s'. Expected %s, got %s."
            if type(set_dim) is type:
                set_dim = set_dim.__name__
                msg += '\n    ' + '\n    '.join(textwrap.wrap(textwrap.dedent("\n                                Slicing components relies on knowing the\n                                underlying set dimensionality (even if the\n                                dimensionality is None).  The underlying\n                                component set ('%s') dimensionality has not been\n                                determined (likely because it is an empty Set).\n                                You can avoid this error by specifying the Set\n                                dimensionality (with the 'dimen=' keyword)." % (self.index_set(),)).strip()))
            raise IndexError(msg % (IndexedComponent_slice._getitem_args_to_str(list(idx)), self.name, set_dim, slice_dim))
        return IndexedComponent_slice(self, fixed, sliced, ellipsis)
    elif len(idx) == len(fixed):
        if len(idx) == 1:
            return fixed[0]
        else:
            return tuple((fixed[i] for i in range(len(idx))))
    else:
        raise DeveloperError(f"Unknown problem encountered when trying to retrieve index '{orig_idx}' for component '{self.name}'")