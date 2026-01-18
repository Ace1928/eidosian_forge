import json
import logging
import warnings
from json import JSONEncoder
from typing import (
from pyquil.experiment._calibration import CalibrationMethod
from pyquil.experiment._memory import (
from pyquil.experiment._program import (
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting, _OneQState, TensorProductState
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.gates import RESET
from pyquil.paulis import PauliTerm, is_identity
from pyquil.quil import Program
from pyquil.quilbase import Reset, ResetQubit
def _pauli_to_product_state(in_state: PauliTerm) -> TensorProductState:
    """
    Convert a Pauli term to a TensorProductState.
    """
    if is_identity(in_state):
        return TensorProductState()
    else:
        return TensorProductState([_OneQState(label=pauli_label, index=0, qubit=cast(int, qubit)) for qubit, pauli_label in in_state._ops.items()])