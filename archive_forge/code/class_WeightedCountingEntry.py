from time import time as gettime
class WeightedCountingEntry(object):

    def __init__(self, value, oneweight):
        self._value = value
        self.weight = self._oneweight = oneweight

    def value(self):
        self.weight += self._oneweight
        return self._value
    value = property(value)