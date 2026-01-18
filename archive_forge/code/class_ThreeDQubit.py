from typing import List
from numpy import sqrt
import numpy as np
import cirq
class ThreeDQubit(cirq.ops.Qid):
    """A qubit in 3d.

    ThreeDQubits use z-y-x ordering:

        ThreeDQubit(0, 0, 0) < ThreeDQubit(1, 0, 0)
        < ThreeDQubit(0, 1, 0) < ThreeDQubit(1, 1, 0)
        < ThreeDQubit(0, 0, 1) < ThreeDQubit(1, 0, 1)
        < ThreeDQubit(0, 1, 1) < ThreeDQubit(1, 1, 1)
    """

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def _comparison_key(self):
        return (round(self.z, 15), round(self.y, 15), round(self.x, 15))

    @property
    def dimension(self) -> int:
        return 2

    def distance(self, other: cirq.ops.Qid) -> float:
        """Returns the distance between two qubits in 3d."""
        if not isinstance(other, ThreeDQubit):
            raise TypeError(f'Can compute distance to another ThreeDQubit, but {other}')
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    @staticmethod
    def cube(diameter: int, x0: float=0, y0: float=0, z0: float=0) -> List['ThreeDQubit']:
        """Returns a cube of ThreeDQubits.

        Args:
            diameter: Length of a side of the square.
            x0: x-coordinate of the first qubit.
            y0: y-coordinate of the first qubit
            z0: z-coordinate of the first qubit.

        Returns:
            A list of ThreeDQubits filling in a square grid
        """
        return ThreeDQubit.parallelep(diameter, diameter, diameter, x0=x0, y0=y0, z0=z0)

    @staticmethod
    def parallelep(rows: int, cols: int, lays: int, x0: float=0, y0: float=0, z0: float=0) -> List['ThreeDQubit']:
        """Returns a parallelepiped of ThreeDQubits.

        Args:
            rows: Number of rows in the parallelepiped.
            cols: Number of columns in the parallelepiped.
            lays: Number of layers in the parallelepiped.
            x0: x-coordinate of the first qubit.
            y0: y-coordinate of the first qubit.
            z0: z-coordinate of the first qubit.

        Returns:
            A list of ThreeDQubits filling in a 3d grid
        """
        return [ThreeDQubit(x0 + x, y0 + y, z0 + z) for z in range(lays) for y in range(cols) for x in range(rows)]

    def __repr__(self):
        return f'pasqal.ThreeDQubit({self.x}, {self.y}, {self.z})'

    def __str__(self):
        return f'({self.x}, {self.y}, {self.z})'

    def _json_dict_(self):
        return cirq.protocols.obj_to_dict_helper(self, ['x', 'y', 'z'])