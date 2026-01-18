from __future__ import unicode_literals
import itertools
import struct
class _TotalOrderingMixin(object):
    __slots__ = ()

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return NotImplemented
        return not equal

    def __lt__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        less = self.__lt__(other)
        if less is NotImplemented or not less:
            return self.__eq__(other)
        return less

    def __gt__(self, other):
        less = self.__lt__(other)
        if less is NotImplemented:
            return NotImplemented
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return NotImplemented
        return not (less or equal)

    def __ge__(self, other):
        less = self.__lt__(other)
        if less is NotImplemented:
            return NotImplemented
        return not less