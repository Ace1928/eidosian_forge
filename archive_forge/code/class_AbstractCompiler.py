from abc import ABC, abstractmethod
from dataclasses import dataclass
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Union
from pyquil._memory import Memory
from pyquil._version import pyquil_version
from pyquil.api._compiler_client import CompilerClient, CompileToNativeQuilRequest
from pyquil.external.rpcq import compiler_isa_to_target_quantum_processor
from pyquil.parser import parse_program
from pyquil.paulis import PauliTerm
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.quil import Program
from pyquil.quilatom import ExpressionDesignator, MemoryReference
from pyquil.quilbase import Gate
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import NativeQuilMetadata, ParameterAref, ParameterSpec
class AbstractCompiler(ABC):
    """The abstract interface for a compiler."""

    def __init__(self, *, quantum_processor: AbstractQuantumProcessor, timeout: float, client_configuration: Optional[QCSClientConfiguration]) -> None:
        self.quantum_processor = quantum_processor
        self._timeout = timeout
        self._client_configuration = client_configuration or QCSClientConfiguration.load()
        self._compiler_client = CompilerClient(client_configuration=self._client_configuration, request_timeout=timeout)

    def get_version_info(self) -> Dict[str, Any]:
        """
        Return version information for this compiler and its dependencies.

        :return: Dictionary of version information.
        """
        return {'quilc': self._compiler_client.get_version()}

    def quil_to_native_quil(self, program: Program, *, protoquil: Optional[bool]=None) -> Program:
        """
        Compile an arbitrary quil program according to the ISA of ``self.quantum_processor``.

        :param program: Arbitrary quil to compile
        :param protoquil: Whether to restrict to protoquil (``None`` means defer to server)
        :return: Native quil and compiler metadata
        """
        self._connect()
        compiler_isa = self.quantum_processor.to_compiler_isa()
        request = CompileToNativeQuilRequest(program=program.out(calibrations=False), target_quantum_processor=compiler_isa_to_target_quantum_processor(compiler_isa), protoquil=protoquil)
        response = self._compiler_client.compile_to_native_quil(request)
        nq_program = parse_program(response.native_program)
        nq_program.native_quil_metadata = None if response.metadata is None else NativeQuilMetadata(final_rewiring=response.metadata.final_rewiring, gate_depth=response.metadata.gate_depth, gate_volume=response.metadata.gate_volume, multiqubit_gate_depth=response.metadata.multiqubit_gate_depth, program_duration=response.metadata.program_duration, program_fidelity=response.metadata.program_fidelity, topological_swaps=response.metadata.topological_swaps, qpu_runtime_estimation=response.metadata.qpu_runtime_estimation)
        nq_program.num_shots = program.num_shots
        nq_program._calibrations = program.calibrations
        nq_program._memory = program._memory.copy()
        return nq_program

    def _connect(self) -> None:
        try:
            _check_quilc_version(self._compiler_client.get_version())
        except TimeoutError:
            raise QuilcNotRunning(f'Request to quilc at {self._compiler_client.base_url} timed out. This could mean that quilc is not running, is not reachable, or is responding slowly. See the Troubleshooting Guide: {{DOCS_URL}}/troubleshooting.html')

    @abstractmethod
    def native_quil_to_executable(self, nq_program: Program) -> QuantumExecutable:
        """
        Compile a native quil program to a binary executable.

        :param nq_program: Native quil to compile
        :return: An (opaque) binary executable
        """

    def reset(self) -> None:
        """
        Reset the state of the this compiler.
        """
        pass