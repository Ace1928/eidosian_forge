import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
class FailHash:

    def __hash__(self):
        raise ValueError('injected failure')