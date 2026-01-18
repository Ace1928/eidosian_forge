import abc
import functools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Union
from typing_extensions import Self
from cirq import ops, protocols
class LineQubit(_BaseLineQid):
    """A qubit on a 1d lattice with nearest-neighbor connectivity.

    LineQubits have a single attribute, and integer coordinate 'x', which
    identifies the qubits location on the line. LineQubits are ordered by
    this integer.

    One can construct new `cirq.LineQubit`s by adding or subtracting integers:

    >>> cirq.LineQubit(1) + 3
    cirq.LineQubit(4)

    >>> cirq.LineQubit(2) - 1
    cirq.LineQubit(1)

    """
    _dimension = 2

    def __init__(self, x: int) -> None:
        """Initializes a line qubit at the given x coordinate.

        Args:
            x: The x coordinate.
        """
        self._x = x

    def _with_x(self, x: int) -> 'LineQubit':
        return LineQubit(x)

    def _cmp_tuple(self):
        cls = LineQid if type(self) is LineQubit else type(self)
        return (cls.__name__, repr(cls), self._comparison_key(), self._dimension)

    @staticmethod
    def range(*range_args) -> List['LineQubit']:
        """Returns a range of line qubits.

        Args:
            *range_args: Same arguments as python's built-in range method.

        Returns:
            A list of line qubits.
        """
        return [LineQubit(i) for i in range(*range_args)]

    def __repr__(self) -> str:
        return f'cirq.LineQubit({self._x})'

    def __str__(self) -> str:
        return f'q({self._x})'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=(f'{self._x}',))

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['x'])