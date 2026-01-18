import warnings
from typing import Dict, List, Union, Optional, Set, cast, Iterable, Sequence, Tuple
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._qvm import (
from pyquil.api._qvm_client import (
from pyquil.gates import MOVE
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program, percolate_declares
from pyquil.quilatom import MemoryReference
from pyquil.wavefunction import Wavefunction
def _expectation_request(self, *, prep_prog: Program, operator_programs: Optional[Iterable[Program]]) -> MeasureExpectationRequest:
    if operator_programs is None:
        operator_programs = [Program()]
    if not isinstance(prep_prog, Program):
        raise TypeError(f'prep_prog must be a Program object, got type {type(prep_prog)}')
    return MeasureExpectationRequest(prep_program=prep_prog.out(calibrations=False), pauli_operators=[x.out(calibrations=False) for x in operator_programs], seed=self.random_seed)