from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, SupportsFloat
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
class ApproximateCircuit(QuantumCircuit, ABC):
    """A base class that represents an approximate circuit."""

    def __init__(self, num_qubits: int, name: Optional[str]=None) -> None:
        """
        Args:
            num_qubits: number of qubit this circuit will span.
            name: a name of the circuit.
        """
        super().__init__(num_qubits, name=name)

    @property
    @abstractmethod
    def thetas(self) -> np.ndarray:
        """
        The property is not implemented and raises a ``NotImplementedException`` exception.

        Returns:
            a vector of parameters of this circuit.
        """
        raise NotImplementedError

    @abstractmethod
    def build(self, thetas: np.ndarray) -> None:
        """
        Constructs this circuit out of the parameters(thetas). Parameter values must be set before
            constructing the circuit.

        Args:
            thetas: a vector of parameters to be set in this circuit.
        """
        raise NotImplementedError