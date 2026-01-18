import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
class SingleQubitOp(cirq.Operation):

    @property
    def qubits(self) -> Tuple[cirq.Qid, ...]:
        return ()

    def with_qubits(self, *new_qubits: cirq.Qid):
        pass

    def __str__(self):
        return 'Op(q2)'

    def _has_mixture_(self):
        return True