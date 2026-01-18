import abc
import copy
import functools
import itertools
import warnings
from enum import IntEnum
from typing import List
import numpy as np
from numpy.linalg import multi_dot
from scipy.sparse import coo_matrix, eye, kron
import pennylane as qml
from pennylane.math import expand_matrix
from pennylane.queuing import QueuingManager
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .utils import pauli_eigs
from .pytrees import register_pytree
class StatePrepBase(Operation):
    """An interface for state-prep operations."""
    grad_method = None

    @abc.abstractmethod
    def state_vector(self, wire_order=None):
        """
        Returns the initial state vector for a circuit given a state preparation.

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels
                from the operator's wires

        Returns:
            array: A state vector for all wires in a circuit
        """

    def label(self, decimals=None, base_label=None, cache=None):
        return '|Ψ⟩'