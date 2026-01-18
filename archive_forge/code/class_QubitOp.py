from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class QubitOp(FixedQids):

    @property
    def qubits(self):
        return cirq.LineQubit.range(2)