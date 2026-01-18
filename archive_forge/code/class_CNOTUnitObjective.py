from __future__ import annotations
import typing
from abc import ABC
import numpy as np
from numpy import linalg as la
from .approximate import ApproximatingObjective
from .elementary_operations import ry_matrix, rz_matrix, place_unitary, place_cnot, rx_matrix
class CNOTUnitObjective(ApproximatingObjective, ABC):
    """
    A base class for a problem definition based on CNOT unit. This class may have different
    subclasses for objective and gradient computations.
    """

    def __init__(self, num_qubits: int, cnots: np.ndarray) -> None:
        """
        Args:
            num_qubits: number of qubits.
            cnots: a CNOT structure to be used in the optimization procedure.
        """
        super().__init__()
        self._num_qubits = num_qubits
        self._cnots = cnots
        self._num_cnots = cnots.shape[1]

    @property
    def num_cnots(self):
        """
        Returns:
            A number of CNOT units to be used by the approximate circuit.
        """
        return self._num_cnots

    @property
    def num_thetas(self):
        """
        Returns:
            Number of parameters (angles) of rotation gates in this circuit.
        """
        return 3 * self._num_qubits + 4 * self._num_cnots