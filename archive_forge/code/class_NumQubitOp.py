from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class NumQubitOp(FixedQids):

    @property
    def qubits(self):
        return cirq.LineQubit.range(3)

    def _num_qubits_(self):
        return 3