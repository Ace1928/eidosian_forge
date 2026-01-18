from collections import defaultdict
from collections.abc import Sequence
import types as pytypes
import weakref
import threading
import contextlib
import operator
import numba
from numba.core import types, errors
from numba.core.typeconv import Conversion, rules
from numba.core.typing import templates
from numba.core.utils import order_by_target_specificity
from .typeof import typeof, Purpose
from numba.core import utils
def _get_attribute_templates(self, typ):
    """
        Get matching AttributeTemplates for the Numba type.
        """
    if typ in self._attributes:
        for attrinfo in self._attributes[typ]:
            yield attrinfo
    else:
        for cls in type(typ).__mro__:
            if cls in self._attributes:
                for attrinfo in self._attributes[cls]:
                    yield attrinfo