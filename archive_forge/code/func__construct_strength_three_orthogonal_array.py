import itertools
import warnings
from math import log, pi
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast
import networkx as nx
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._compiler import AbstractCompiler, QVMCompiler
from pyquil.api._qam import QAM
from pyquil.api._qpu import QPU
from pyquil.api._quantum_computer import QuantumComputer as QuantumComputerV3
from pyquil.api._quantum_computer import get_qc as get_qc_v3, QuantumExecutable
from pyquil.api._qvm import QVM
from pyquil.experiment._main import Experiment
from pyquil.experiment._memory import merge_memory_map_lists
from pyquil.experiment._result import ExperimentResult, bitstrings_to_expectations
from pyquil.experiment._setting import ExperimentSetting
from pyquil.gates import MEASURE, RX
from pyquil.noise import NoiseModel, decoherence_noise_with_asymmetric_ro
from pyquil.paulis import PauliTerm
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import AbstractQuantumProcessor, NxQuantumProcessor
from pyquil.quil import Program, validate_supported_quil
from pyquil.quilatom import qubit_index
from ._qam import StatefulQAM
def _construct_strength_three_orthogonal_array(num_qubits: int) -> np.ndarray:
    """
    Given a number of qubits this function returns an Orthogonal Array (OA)
    on 'n' qubits where n is the next power of two relative to num_qubits.

    Specifically it returns the OA(2n, n, 2, 3).

    The parameters of the OA(N, k, s, t) are interpreted as
    N: Number of rows, level combinations or runs
    k: Number of columns, constraints or factors
    s: Number of symbols or levels
    t: Strength

    See [OATA] for more details.

    [OATA] Orthogonal Arrays: theory and applications
           Hedayat, Sloane, Stufken
           Springer Science & Business Media, 2012.
           https://dx.doi.org/10.1007/978-1-4612-1478-6

    :param num_qubits: minimum number of qubits the OA should run on.
    :return: A numpy array representing the OA with shape N by k
    """
    num_qubits_power_of_2 = _next_power_of_2(num_qubits)
    H = hadamard(num_qubits_power_of_2)
    Hfold = np.concatenate((H, -H), axis=0)
    orthogonal_array = cast(np.ndarray, ((Hfold + 1) / 2).astype(int))
    return orthogonal_array