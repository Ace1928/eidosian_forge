from __future__ import absolute_import, print_function, division
import operator
from functools import partial
from petl.compat import text_type, binary_type, numeric_types
def _itemgetter_with_default(*args):
    """ itemgetter compatible with `operator.itemgetter` behavior, filling missing
    values with default instead of raising IndexError or KeyError """

    def _get_default(obj, item, default):
        try:
            return obj[item]
        except (IndexError, KeyError):
            return default
    if len(args) == 1:
        return partial(_get_default, item=args[0], default=None)
    return lambda obj: tuple((_get_default(obj, item=item, default=None) for item in args))