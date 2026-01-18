import math
from collections.abc import Sequence
from pyomo.common.numeric_types import check_if_numeric_type
class RangeProduct(object):
    """A range-like object for representing the cross product of ranges"""
    __slots__ = ('range_lists',)

    def __init__(self, range_lists):
        self.range_lists = range_lists
        assert range_lists.__class__ is list
        for subrange in range_lists:
            assert subrange.__class__ is list

    def __str__(self):
        return '<' + ', '.join((str(tuple(_)) if len(_) > 1 else str(_[0]) for _ in self.range_lists)) + '>'
    __repr__ = __str__

    def __eq__(self, other):
        return isinstance(other, RangeProduct) and self.range_difference([other]) == [] and (other.range_difference([self]) == [])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, value):
        if not isinstance(value, Sequence):
            return False
        if len(value) != len(self.range_lists):
            return False
        return all((any((val in rng for rng in rng_list)) for val, rng_list in zip(value, self.range_lists)))

    def __getstate__(self):
        """
        Retrieve the state of this object as a dictionary.

        This method must be defined because this class uses slots.
        """
        state = {}
        for i in RangeProduct.__slots__:
            state[i] = getattr(self, i)
        return state

    def __setstate__(self, state):
        """
        Set the state of this object using values from a state dictionary.

        This method must be defined because this class uses slots.
        """
        for key, val in state.items():
            object.__setattr__(self, key, val)

    def isdiscrete(self):
        return all((all((rng.isdiscrete() for rng in rng_list)) for rng_list in self.range_lists))

    def isfinite(self):
        return all((all((rng.isfinite() for rng in rng_list)) for rng_list in self.range_lists))

    def isdisjoint(self, other):
        if type(other) is AnyRange:
            return False
        if type(other) is not RangeProduct:
            return True
        if len(other.range_lists) != len(self.range_lists):
            return True
        for s, o in zip(self.range_lists, other.range_lists):
            if all((s_rng.isdisjoint(o_rng) for s_rng in s for o_rng in o)):
                return True
        return False

    def issubset(self, other):
        if type(other) is AnyRange:
            return True
        return not any((_ for _ in self.range_difference([other])))

    def range_difference(self, other_ranges):
        ans = [self]
        N = len(self.range_lists)
        for other in other_ranges:
            if type(other) is AnyRange:
                return []
            if type(other) is not RangeProduct or len(other.range_lists) != N:
                continue
            tmp = []
            for rp in ans:
                if rp.isdisjoint(other):
                    tmp.append(rp)
                    continue
                for dim in range(N):
                    remainder = []
                    for r in rp.range_lists[dim]:
                        remainder.extend(r.range_difference(other.range_lists[dim]))
                    if remainder:
                        tmp.append(RangeProduct(list(rp.range_lists)))
                        tmp[-1].range_lists[dim] = remainder
            ans = tmp
        return ans

    def range_intersection(self, other_ranges):
        ans = list(self.range_lists)
        N = len(self.range_lists)
        for other in other_ranges:
            if type(other) is AnyRange:
                continue
            if type(other) is not RangeProduct or len(other.range_lists) != N:
                return []
            for dim in range(N):
                tmp = []
                for r in ans[dim]:
                    tmp.extend(r.range_intersection(other.range_lists[dim]))
                if not tmp:
                    return []
                ans[dim] = tmp
        return [RangeProduct(ans)]