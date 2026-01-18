import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Type, Union
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.api import QAM, QuantumExecutable, QAMExecutionResult
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.quil import Program
from pyquil.quilatom import Label, LabelPlaceholder, MemoryReference
from pyquil.quilbase import (
class AbstractQuantumSimulator(ABC):

    @abstractmethod
    def __init__(self, n_qubits: int, rs: RandomState):
        """
        Initialize.

        :param n_qubits: Number of qubits to simulate.
        :param rs: a RandomState (shared with the owning :py:class:`PyQVM`) for
            doing anything stochastic.
        """

    @abstractmethod
    def do_gate(self, gate: Gate) -> 'AbstractQuantumSimulator':
        """
        Perform a gate.

        :return: ``self`` to support method chaining.
        """

    @abstractmethod
    def do_gate_matrix(self, matrix: np.ndarray, qubits: Sequence[int]) -> 'AbstractQuantumSimulator':
        """
        Apply an arbitrary unitary; not necessarily a named gate.

        :param matrix: The unitary matrix to apply. No checks are done
        :param qubits: A list of qubits to apply the unitary to.
        :return: ``self`` to support method chaining.
        """

    def do_program(self, program: Program) -> 'AbstractQuantumSimulator':
        """
        Perform a sequence of gates contained within a program.

        :param program: The program
        :return: self
        """
        for gate in program:
            if not isinstance(gate, Gate):
                raise ValueError('Can only compute the simulate a program composed of `Gate`s')
            self.do_gate(gate)
        return self

    @abstractmethod
    def do_measurement(self, qubit: int) -> int:
        """
        Measure a qubit and collapse the wavefunction

        :return: The measurement result. A 1 or a 0.
        """

    @abstractmethod
    def expectation(self, operator: Union[PauliTerm, PauliSum]) -> complex:
        """
        Compute the expectation of an operator.

        :param operator: The operator
        :return: The operator's expectation value
        """

    @abstractmethod
    def reset(self) -> 'AbstractQuantumSimulator':
        """
        Reset the wavefunction to the ``|000...00>`` state.

        :return: ``self`` to support method chaining.
        """

    @abstractmethod
    def sample_bitstrings(self, n_samples: int) -> np.ndarray:
        """
        Sample bitstrings from the current state.

        :param n_samples: The number of bitstrings to sample
        :return: A numpy array of shape (n_samples, n_qubits)
        """

    @abstractmethod
    def do_post_gate_noise(self, noise_type: str, noise_prob: float, qubits: List[int]) -> 'AbstractQuantumSimulator':
        """
        Apply noise that happens after each gate application.

        WARNING! This is experimental and the signature of this interface will likely change.

        :param noise_type: The name of the noise type
        :param noise_prob: The probability of that noise happening
        :param qubits: Apply noise to these qubits.
        :return: ``self`` to support method chaining
        """