import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
class AllEqual:
    __hash__ = None

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False