import pytest
import cirq
@cirq.value_equality(manual_cls=True)
class MasqueradePositiveD:

    def __init__(self, x):
        self.x = x

    def _value_equality_values_(self):
        return self.x

    def _value_equality_values_cls_(self):
        return BasicD if self.x > 0 else MasqueradePositiveD