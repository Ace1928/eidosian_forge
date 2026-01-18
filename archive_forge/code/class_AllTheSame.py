import unittest
from traits.util.weakiddict import WeakIDDict, WeakIDKeyDict
class AllTheSame(object):

    def __hash__(self):
        return 42

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __ne__(self, other):
        return not self.__eq__(other)