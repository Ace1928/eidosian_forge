from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
class CustomXGateFamily(cirq.GateFamily):
    """Accepts all integer powers of CustomXPowGate"""

    def __init__(self) -> None:
        super().__init__(gate=CustomXPowGate, name='CustomXGateFamily', description='Accepts all integer powers of CustomXPowGate')

    def _predicate(self, g: cirq.Gate) -> bool:
        """Checks whether gate instance `g` belongs to this GateFamily."""
        if not super()._predicate(g) or cirq.is_parameterized(g):
            return False
        exp = cast(CustomXPowGate, g).exponent
        return int(exp) == exp

    def __repr__(self):
        return 'cirq.ops.gateset_test.CustomXGateFamily()'