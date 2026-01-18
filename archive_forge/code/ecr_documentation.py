from math import sqrt
import numpy as np
from qiskit.circuit._utils import with_gate_array
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from .rzx import RZXGate
from .x import XGate
Return inverse ECR gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            ECRGate: inverse gate (self-inverse).
        