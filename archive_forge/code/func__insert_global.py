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
def _insert_global(self, gv, gty):
    """
        Register type *gty* for value *gv*.  Only a weak reference
        to *gv* is kept, if possible.
        """

    def on_disposal(wr, pop=self._globals.pop):
        pop(wr)
    try:
        gv = weakref.ref(gv, on_disposal)
    except TypeError:
        pass
    self._globals[gv] = gty