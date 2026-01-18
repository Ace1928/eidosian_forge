import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
class InconsistentNeImplementation:

    def __init__(self):
        self.x = 1

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.x == other.x

    def __ne__(self, other):
        return NotImplemented

    def __hash__(self):
        return hash(self.x)