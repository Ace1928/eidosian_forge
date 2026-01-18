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
def _one_q_pauli_prep(label: str, index: int, qubit: QubitDesignator) -> Program:
    """Prepare the index-th eigenstate of the pauli operator given by label."""
    if index not in [0, 1]:
        raise ValueError(f'Bad Pauli index: {index}')
    if label == 'X':
        if index == 0:
            return Program(RY(pi / 2, qubit))
        else:
            return Program(RY(-pi / 2, qubit))
    elif label == 'Y':
        if index == 0:
            return Program(RX(-pi / 2, qubit))
        else:
            return Program(RX(pi / 2, qubit))
    elif label == 'Z':
        if index == 0:
            return Program()
        else:
            return Program(RX(pi, qubit))
    raise ValueError(f'Bad Pauli label: {label}')