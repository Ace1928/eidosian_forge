import pytest
import sympy
import cirq
class RMulReturnsNotImplemented:

    def __rmul__(self, other):
        return NotImplemented