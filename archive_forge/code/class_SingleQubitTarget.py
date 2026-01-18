from typing import List, Union, Type, cast, TYPE_CHECKING
from enum import Enum
import numpy as np
from cirq import ops, transformers, protocols, linalg
from cirq.type_workarounds import NotImplementedType
class SingleQubitTarget(Enum):
    SINGLE_QUBIT_CLIFFORDS = 1
    PAULI_STRING_PHASORS_AND_CLIFFORDS = 2
    PAULI_STRING_PHASORS = 3