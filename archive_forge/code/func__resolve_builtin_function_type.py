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
def _resolve_builtin_function_type(self, func, args, kws):
    if func in self._functions:
        defns = self._functions[func]
        for defn in defns:
            for support_literals in [True, False]:
                if support_literals:
                    res = defn.apply(args, kws)
                else:
                    fixedargs = [types.unliteral(a) for a in args]
                    res = defn.apply(fixedargs, kws)
                if res is not None:
                    return res