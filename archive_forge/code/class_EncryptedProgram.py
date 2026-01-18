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
@dataclass
class EncryptedProgram:
    """
    Encrypted binary, executable on a QPU.
    """
    program: str
    'String representation of an encrypted Quil program.'
    memory_descriptors: Dict[str, ParameterSpec]
    "Descriptors for memory executable's regions, mapped by name."
    ro_sources: Dict[MemoryReference, str]
    'Readout sources, mapped by memory reference.'
    recalculation_table: Dict[ParameterAref, ExpressionDesignator]
    'A mapping from memory references to the original gate arithmetic.'
    _memory: Memory
    'Memory values (parameters) to be sent with the program.'

    def copy(self) -> 'EncryptedProgram':
        """
        Return a deep copy of this EncryptedProgram.
        """
        return dataclasses.replace(self, _memory=self._memory.copy())

    def write_memory(self, *, region_name: str, value: Union[int, float, Sequence[int], Sequence[float]], offset: Optional[int]=None) -> 'EncryptedProgram':
        self._memory._write_value(parameter=ParameterAref(name=region_name, index=offset or 0), value=value)
        return self