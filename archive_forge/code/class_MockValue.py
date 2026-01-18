import fractions
import pytest
import cirq
class MockValue:

    def __init__(self, val, eq, ne, lt, gt, le, ge):
        self.val = val
        self._eq = eq
        self._ne = ne
        self._lt = lt
        self._gt = gt
        self._le = le
        self._ge = ge
    __hash__ = None

    def __eq__(self, other):
        return self._eq(self, other)

    def __ne__(self, other):
        return self._ne(self, other)

    def __lt__(self, other):
        return self._lt(self, other)

    def __gt__(self, other):
        return self._gt(self, other)

    def __le__(self, other):
        return self._le(self, other)

    def __ge__(self, other):
        return self._ge(self, other)

    def __repr__(self):
        return f'MockValue(val={self.val!r}, ...)'