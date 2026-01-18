import re
from itertools import product
import numpy as np
import copy
from typing import (
from pyquil.quilatom import (
from .quil import Program
from .gates import H, RZ, RX, CNOT, X, PHASE, QUANTUM_GATES
from numbers import Number, Complex
from collections import OrderedDict
import warnings
def _multiply_factor(self, factor: str, index: PauliTargetDesignator) -> 'PauliTerm':
    new_term = PauliTerm('I', 0)
    new_coeff = self.coefficient
    new_ops = self._ops.copy()
    ops = self[index] + factor
    new_op = PAULI_PROD[ops]
    if new_op != 'I':
        new_ops[index] = new_op
    else:
        del new_ops[index]
    new_coeff *= PAULI_COEFF[ops]
    new_term._ops = new_ops
    new_term.coefficient = new_coeff
    return new_term