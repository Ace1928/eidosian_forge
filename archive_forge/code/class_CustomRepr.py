import numpy as np
import pytest
import cirq
class CustomRepr:
    setup_code = 'class CustomRepr:\n            def __init__(self, eq_val):\n                self.eq_val = eq_val\n            def __pow__(self, exponent):\n                return self\n        '

    def __init__(self, eq_val, repr_str: str):
        self.eq_val = eq_val
        self.repr_str = repr_str

    def __eq__(self, other):
        return self.eq_val == getattr(other, 'eq_val', None)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return self.repr_str