import abc
import numbers
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, linalg, value
from cirq._compat import proper_repr
from cirq.ops import raw_types, identity, pauli_gates, global_phase_op, pauli_string
from cirq.type_workarounds import NotImplementedType
Returns a copy with possibly modified contents.

        Args:
            coefficient: The new coefficient value. If not specified, defaults
                to the current `coefficient` value.
            pauli_mask: The new `pauli_mask` value. If not specified, defaults
                to the current pauli mask value.

        Returns:
            A copied instance.
        