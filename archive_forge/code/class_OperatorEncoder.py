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
class OperatorEncoder(JSONEncoder):

    def default(self, o: Any) -> Any:
        if isinstance(o, ExperimentSetting):
            return o.serializable()
        if isinstance(o, Experiment):
            return o.serializable()
        if isinstance(o, ExperimentResult):
            return o.serializable()
        return o