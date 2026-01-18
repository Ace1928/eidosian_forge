from typing import List, cast, Optional, Union, Dict, Any
import functools
from math import sqrt
import httpx
import numpy as np
import networkx as nx
import cirq
from pyquil.quantum_processor import QCSQuantumProcessor
from qcs_api_client.models import InstructionSetArchitecture
from qcs_api_client.operations.sync import get_instruction_set_architecture
from cirq_rigetti._qcs_api_client_decorator import _provide_default_client
class OctagonalQubit(cirq.ops.Qid):
    """A cirq.Qid supporting Octagonal indexing."""

    def __init__(self, octagon_position: int):
        """Initializes an `OctagonalQubit` using indices 0-7.
              4  - 3
            /        \\
          5           2
          |           |
          6           1
            \\       /
              7 - 0

        Args:
            octagon_position: Position within octagon, indexed as pictured above.

        Returns:
            The initialized `OctagonalQubit`.

        Raises:
            ValueError: If the position specified is greater than 7.
        """
        if octagon_position >= 8:
            raise ValueError(f'OctagonQubit must be less than 8, received {octagon_position}')
        self._octagon_position = octagon_position
        self.index = octagon_position

    @property
    def octagon_position(self):
        return self._octagon_position

    def _comparison_key(self):
        return self.index

    @property
    def dimension(self) -> int:
        return 2

    def distance(self, other: cirq.Qid) -> float:
        """Returns the distance between two qubits.

        Args:
            other: An OctagonalQubit to which we are measuring distance.

        Returns:
            The distance between two qubits.

        Raises:
            TypeError: other qubit must be OctagonalQubit.
        """
        if type(other) != OctagonalQubit:
            raise TypeError('can only measure distance from other Octagonal qubits')
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    @property
    def x(self) -> float:
        """Returns the horizontal position of the qubit, assuming each side of
        the octagon has length 1.

        Returns:
            The horizontal position of the qubit.

        Raises:
            ValueError: Octagon position is invalid.
        """
        if self.octagon_position in {5, 6}:
            return 0
        if self.octagon_position in {4, 7}:
            return 1 / sqrt(2)
        if self.octagon_position in {0, 3}:
            return 1 + 1 / sqrt(2)
        if self.octagon_position in {1, 2}:
            return 1 + sqrt(2)
        raise ValueError(f'invalid octagon position {self.octagon_position}')

    @property
    def y(self) -> float:
        """Returns the vertical position of the qubit, assuming each side of
        the octagon has length 1. The y-axis is oriented downwards.

        Returns:
            The vertical position of the qubit.

        Raises:
            ValueError: Octagon position is invalid.
        """
        if self.octagon_position in {3, 4}:
            return 0
        if self.octagon_position in {2, 5}:
            return 1 / sqrt(2)
        if self.octagon_position in {1, 6}:
            return 1 + 1 / sqrt(2)
        if self.octagon_position in {0, 7}:
            return 1 + sqrt(2)
        raise ValueError(f'invalid octagon position {self.octagon_position}')

    @property
    def z(self) -> int:
        """Because this is a 2-dimensional qubit, this will always be 0.

        Returns:
            Zero.
        """
        return 0

    def __repr__(self):
        return f'cirq_rigetti.OctagonalQubit(octagon_position={self.octagon_position})'

    def _json_dict_(self):
        return {'octagon_position': self.octagon_position}