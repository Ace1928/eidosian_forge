import pytest
import sympy
import cirq
class MulSevenRMulEight:

    def __mul__(self, other):
        return 7

    def __rmul__(self, other):
        return 8