import logging
from math import pi
from numbers import Complex
from typing import Callable, Generator, List, Mapping, Tuple, Optional, cast
import numpy as np
from pyquil.api import QuantumComputer
from pyquil.quil import Program
from pyquil.quilatom import QubitDesignator
from pyquil.experiment._group import (
from pyquil.experiment._main import (
from pyquil.experiment._result import ExperimentResult, ratio_variance
from pyquil.experiment._setting import (
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.gates import RESET, RX, RY, RZ, X
from pyquil.paulis import is_identity
from pyquil.quil import Program
from pyquil.quilatom import QubitDesignator
def _ops_bool_to_prog(ops_bool: Tuple[bool], qubits: List[int]) -> Program:
    """
    :param ops_bool: tuple of booleans specifying the operation to be carried out on `qubits`
    :param qubits: list specifying the qubits to be carried operations on
    :return: Program with the operations specified in `ops_bool` on the qubits specified in
        `qubits`
    """
    assert len(ops_bool) == len(qubits), 'Mismatch of qubits and operations'
    prog = Program()
    for i, op_bool in enumerate(ops_bool):
        if op_bool == 0:
            continue
        elif op_bool == 1:
            prog += Program(X(qubits[i]))
        else:
            raise ValueError('ops_bool should only consist of 0s and/or 1s')
    return prog