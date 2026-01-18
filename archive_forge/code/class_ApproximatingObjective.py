from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, SupportsFloat
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
class ApproximatingObjective(ABC):
    """
    A base class for an optimization problem definition. An implementing class must provide at least
    an implementation of the ``objective`` method. In such case only gradient free optimizers can
    be used. Both method, ``objective`` and ``gradient``, preferable to have in an implementation.
    """

    def __init__(self) -> None:
        self._target_matrix: np.ndarray | None = None

    @abstractmethod
    def objective(self, param_values: np.ndarray) -> SupportsFloat:
        """
        Computes a value of the objective function given a vector of parameter values.

        Args:
            param_values: a vector of parameter values for the optimization problem.

        Returns:
            a float value of the objective function.
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, param_values: np.ndarray) -> np.ndarray:
        """
        Computes a gradient with respect to parameters given a vector of parameter values.

        Args:
            param_values: a vector of parameter values for the optimization problem.

        Returns:
            an array of gradient values.
        """
        raise NotImplementedError

    @property
    def target_matrix(self) -> np.ndarray:
        """
        Returns:
            a matrix being approximated
        """
        return self._target_matrix

    @target_matrix.setter
    def target_matrix(self, target_matrix: np.ndarray) -> None:
        """
        Args:
            target_matrix: a matrix to approximate in the optimization procedure.
        """
        self._target_matrix = target_matrix

    @property
    @abstractmethod
    def num_thetas(self) -> int:
        """

        Returns:
            the number of parameters in this optimization problem.
        """
        raise NotImplementedError