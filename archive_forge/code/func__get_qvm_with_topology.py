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
def _get_qvm_with_topology(*, client_configuration: QCSClientConfiguration, name: str, topology: nx.Graph, noisy: bool, qvm_type: str, compiler_timeout: float, execution_timeout: float) -> QuantumComputer:
    """Construct a QVM with the provided topology.

    :param client_configuration: Client configuration.
    :param name: A name for your quantum computer. This field does not affect behavior of the
        constructed QuantumComputer.
    :param topology: A graph representing the desired qubit connectivity.
    :param noisy: Whether to include a generic noise model. If you want more control over
        the noise model, please construct your own :py:class:`NoiseModel` and use
        :py:func:`_get_qvm_qc` instead of this function.
    :param qvm_type: The type of QVM. Either 'qvm' or 'pyqvm'.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param execution_timeout: Time limit for execution requests, in seconds.
    :return: A pre-configured QuantumComputer
    """
    quantum_processor = NxQuantumProcessor(topology=topology)
    if noisy:
        noise_model: Optional[NoiseModel] = decoherence_noise_with_asymmetric_ro(isa=quantum_processor.to_compiler_isa())
    else:
        noise_model = None
    return _get_qvm_qc(client_configuration=client_configuration, name=name, qvm_type=qvm_type, quantum_processor=quantum_processor, noise_model=noise_model, compiler_timeout=compiler_timeout, execution_timeout=execution_timeout)