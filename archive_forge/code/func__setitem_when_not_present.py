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
def _setitem_when_not_present(self, index, value=_NotSpecified):
    """Perform the fundamental component item creation and storage.

        Components that want to implement a nonstandard storage mechanism
        should override this method.

        Implementations may assume that the index has already been
        validated and is a legitimate entry to add to the _data dict.
        """
    if value is IndexedComponent.Skip:
        return None
    if index is None and (not self.is_indexed()):
        obj = self._data[index] = self
    else:
        obj = self._data[index] = self._ComponentDataClass(component=self)
    obj._index = index
    try:
        if value is not _NotSpecified:
            obj.set_value(value)
    except:
        self._data.pop(index, None)
        raise
    return obj