import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
class Gateless(cirq.Operation):

    @property
    def qubits(self):
        return ()

    def with_qubits(self, *new_qubits):
        return self

    def _has_mixture_(self):
        return True