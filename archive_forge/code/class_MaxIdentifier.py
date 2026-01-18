import functools
import re
import warnings
class MaxIdentifier(object):
    __slots__ = []

    def __repr__(self):
        return 'MaxIdentifier()'

    def __eq__(self, other):
        return isinstance(other, self.__class__)