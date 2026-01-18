import pytest
import sympy
import cirq
class RMulReturnsSix:

    def __rmul__(self, other):
        return 6