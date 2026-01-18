import pytest
import cirq
class ReturnsFive:

    def __pow__(self, exponent) -> int:
        return 5