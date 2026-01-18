import pytest
import cirq
@cirq.value_equality(unhashable=True)
class UnhashableC:

    def __init__(self, x):
        self.x = x

    def _value_equality_values_(self):
        return self.x