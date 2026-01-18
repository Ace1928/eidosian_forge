import unittest
from traits.util.weakiddict import WeakIDDict, WeakIDKeyDict
class WeakreffableInt(object):

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        else:
            return self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)