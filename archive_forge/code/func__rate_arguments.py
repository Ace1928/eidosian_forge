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
def _rate_arguments(self, actualargs, formalargs, unsafe_casting=True, exact_match_required=False):
    """
        Rate the actual arguments for compatibility against the formal
        arguments.  A Rating instance is returned, or None if incompatible.
        """
    if len(actualargs) != len(formalargs):
        return None
    rate = Rating()
    for actual, formal in zip(actualargs, formalargs):
        conv = self.can_convert(actual, formal)
        if conv is None:
            return None
        elif not unsafe_casting and conv >= Conversion.unsafe:
            return None
        elif exact_match_required and conv != Conversion.exact:
            return None
        if conv == Conversion.promote:
            rate.promote += 1
        elif conv == Conversion.safe:
            rate.safe_convert += 1
        elif conv == Conversion.unsafe:
            rate.unsafe_convert += 1
        elif conv == Conversion.exact:
            pass
        else:
            raise Exception('unreachable', conv)
    return rate