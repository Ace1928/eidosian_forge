from typing import Tuple, Optional, List, Union, Generic, TypeVar, Dict
from unittest.mock import create_autospec, Mock
import pytest
from pyquil import Program
from pyquil.quantum_processor import AbstractQuantumProcessor, NxQuantumProcessor
from pyquil.api import QAM, QuantumComputer, QuantumExecutable, QAMExecutionResult, EncryptedProgram
from pyquil.api._abstract_compiler import AbstractCompiler
from qcs_api_client.client._configuration.settings import QCSClientConfigurationSettings
from qcs_api_client.client._configuration import (
import networkx as nx
import cirq
import sympy
import numpy as np
class MockQAM(QAM, Generic[T]):
    _run_count: int
    _mock_results: Dict[str, np.ndarray]

    def __init__(self, *args, **kwargs) -> None:
        self._run_count = 0
        self._mock_results: Dict[str, np.ndarray] = {}

    def execute(self, executable: QuantumExecutable) -> T:
        pass

    def run(self, program: QuantumExecutable) -> QAMExecutionResult:
        raise NotImplementedError

    def get_result(self, execute_response: T) -> QAMExecutionResult:
        raise NotImplementedError