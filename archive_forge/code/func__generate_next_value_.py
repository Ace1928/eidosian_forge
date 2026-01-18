import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
def _generate_next_value_(name, start, count, last_values):
    """
        Generate the next value when not given.

        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the last value assigned or None
        """
    if not count:
        return start if start is not None else 1
    last_value = max(last_values)
    try:
        high_bit = _high_bit(last_value)
    except Exception:
        raise TypeError('invalid flag value %r' % last_value) from None
    return 2 ** (high_bit + 1)