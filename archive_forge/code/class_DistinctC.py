import pytest
import cirq
@cirq.value_equality(distinct_child_types=True)
class DistinctC:

    def __init__(self, x):
        self.x = x

    def _value_equality_values_(self):
        return self.x