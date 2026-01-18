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
def _wavefunction_request(self, *, quil_program: Program) -> GetWavefunctionRequest:
    if not isinstance(quil_program, Program):
        raise TypeError(f'quil_program must be a Program object, got type {type(quil_program)}')
    return GetWavefunctionRequest(program=quil_program.out(calibrations=False), measurement_noise=self.measurement_noise, gate_noise=self.gate_noise, seed=self.random_seed)