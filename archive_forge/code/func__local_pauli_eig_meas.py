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
def _local_pauli_eig_meas(op: str, idx: QubitDesignator) -> Program:
    """
    Generate gate sequence to measure in the eigenbasis of a Pauli operator, assuming
    we are only able to measure in the Z eigenbasis. (Note: The unitary operations of this
    Program are essentially the Hermitian conjugates of those in :py:func:`_one_q_pauli_prep`)

    """
    if op == 'X':
        return Program(RY(-pi / 2, idx))
    elif op == 'Y':
        return Program(RX(pi / 2, idx))
    elif op == 'Z':
        return Program()
    raise ValueError(f'Unknown operation {op}')