import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
class GroupAggregator(object):
    """A function to aggregate the members of a group found by group_by().

    The user-specified function is provided at creation. It is assumed to
    accept the group value list as an argument and return an aggregated value.

    However, on error we will (optionally) fall back to the pre-1.1.1 behaviour
    of assuming that the function expects a tuple containing both the key and
    the value list, and similarly returns a tuple of the key and value. This
    can still give the wrong results if the first group(s) to be aggregated
    have value lists of length exactly 2, but for the most part is backwards
    compatible to 1.1.1.
    """

    def __init__(self, aggregator_func=None, allow_fallback=True):
        self.aggregator = aggregator_func
        self.allow_fallback = allow_fallback
        self._failure_info = None

    def __call__(self, group_item):
        if self.aggregator is None:
            return group_item
        if self._failure_info is None:
            key, value_list = group_item
            try:
                result = self.aggregator(value_list)
            except (exceptions.NoMatchingMethodException, exceptions.NoMatchingFunctionException, IndexError) as exc:
                self._failure_info = exc
            else:
                if not (len(value_list) == 2 and isinstance(result, collections.abc.Sequence) and (not isinstance(result, str)) and (len(result) == 2) and (result[0] == value_list[0])):
                    self.allow_fallback = False
                return (key, result)
        if self.allow_fallback:
            try:
                result = self.aggregator(group_item)
                if len(result) == 2:
                    return result
            except Exception:
                pass
        raise self._failure_info