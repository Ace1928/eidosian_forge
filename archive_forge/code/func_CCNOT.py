from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def CCNOT(control1: QubitDesignator, control2: QubitDesignator, target: QubitDesignator) -> Gate:
    """Produces a doubly-controlled NOT gate::

        CCNOT = [[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0]]

    This gate applies to three qubit arguments to produce the controlled-controlled-not gate
    instruction.

    :param control1: The first control qubit.
    :param control2: The second control qubit.
    :param target: The target qubit. The target qubit has an X-gate applied to it if both control
        qubits are in the excited state.
    :returns: A Gate object.
    """
    qubits = [unpack_qubit(q) for q in (control1, control2, target)]
    return Gate(name='CCNOT', params=[], qubits=qubits)