import pytest
import cirq
class SelfInverse:

    def __pow__(self, exponent) -> 'SelfInverse':
        return self