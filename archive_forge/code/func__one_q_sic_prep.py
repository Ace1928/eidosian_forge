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
def _one_q_sic_prep(index: int, qubit: QubitDesignator) -> Program:
    """Prepare the index-th SIC basis state."""
    if index == 0:
        return Program()
    theta = 2 * np.arccos(1 / np.sqrt(3))
    zx_plane_rotation = Program([RX(-pi / 2, qubit), RZ(theta - pi, qubit), RX(-pi / 2, qubit)])
    if index == 1:
        return zx_plane_rotation
    elif index == 2:
        return zx_plane_rotation + RZ(-2 * pi / 3, qubit)
    elif index == 3:
        return zx_plane_rotation + RZ(2 * pi / 3, qubit)
    raise ValueError(f'Bad SIC index: {index}')