import pytest
import cirq
class ImplementsReversible:

    def __pow__(self, exponent):
        return 6 if exponent == -1 else NotImplemented