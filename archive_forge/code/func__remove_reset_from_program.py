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
def _remove_reset_from_program(program: Program) -> Program:
    """
    Trim the RESET from a program because in measure_observables it is re-added.

    :param program: Program to remove RESET(s) from.
    :return: Trimmed Program.
    """
    p = program.copy_everything_except_instructions()
    for inst in program:
        if not isinstance(inst, (Reset, ResetQubit)):
            p.inst(inst)
    return p