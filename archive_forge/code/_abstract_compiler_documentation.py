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

        Construct a randomized benchmarking experiment on the given qubits, decomposing into
        gateset. If interleaver is not provided, the returned sequence will have the form

            C_1 C_2 ... C_(depth-1) C_inv ,

        where each C is a Clifford element drawn from gateset, C_{< depth} are randomly selected,
        and C_inv is selected so that the entire sequence composes to the identity.  If an
        interleaver G (which must be a Clifford, and which will be decomposed into the native
        gateset) is provided, then the sequence instead takes the form

            C_1 G C_2 G ... C_(depth-1) G C_inv .

        The JSON response is a list of lists of indices, or Nones. In the former case, they are the
        index of the gate in the gateset.

        :param int depth: The number of Clifford gates to include in the randomized benchmarking
         experiment. This is different than the number of gates in the resulting experiment.
        :param list gateset: A list of pyquil gates to decompose the Clifford elements into. These
         must generate the clifford group on the qubits of interest. e.g. for one qubit
         [RZ(np.pi/2), RX(np.pi/2)].
        :param seed: A positive integer used to seed the PRNG.
        :param interleaver: A Program object that encodes a Clifford element.
        :return: A list of pyquil programs. Each pyquil program is a circuit that represents an
         element of the Clifford group. When these programs are composed, the resulting Program
         will be the randomized benchmarking experiment of the desired depth. e.g. if the return
         programs are called cliffords then `sum(cliffords, Program())` will give the randomized
         benchmarking experiment, which will compose to the identity program.
        