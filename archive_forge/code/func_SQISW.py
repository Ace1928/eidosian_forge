from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def SQISW(q1: QubitDesignator, q2: QubitDesignator) -> Gate:
    """Produces a SQISW gate::

        SQiSW = [[1,               0,               0, 0],
                 [0,     1 / sqrt(2),    1j / sqrt(2), 0],
                 [0,    1j / sqrt(2),     1 / sqrt(2), 0],
                 [0,               0,               0, 1]]

    :param q1: Qubit 1.
    :param q2: Qubit 2.
    :returns: A Gate object.
    """
    return Gate(name='SQISW', params=[], qubits=[unpack_qubit(q) for q in (q1, q2)])