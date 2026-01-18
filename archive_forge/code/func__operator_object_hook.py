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
def _operator_object_hook(obj: Mapping[str, Any]) -> Union[Mapping[str, Any], Experiment]:
    if 'type' in obj and obj['type'] in ['Experiment', 'TomographyExperiment']:
        settings = [[ExperimentSetting.from_str(s) for s in stt] for stt in obj['settings']]
        p = Program(obj['program'])
        p.wrap_in_numshots_loop(obj['shots'])
        ex = Experiment(settings=settings, program=p, symmetrization=obj['symmetrization'])
        ex.reset = obj['reset']
        return ex
    return obj