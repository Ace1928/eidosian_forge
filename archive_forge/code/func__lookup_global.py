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
def _lookup_global(self, gv):
    """
        Look up the registered type for global value *gv*.
        """
    try:
        gv = weakref.ref(gv)
    except TypeError:
        pass
    try:
        return self._globals.get(gv, None)
    except TypeError:
        return None