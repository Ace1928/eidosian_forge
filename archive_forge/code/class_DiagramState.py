from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
class DiagramState:
    """
    A representation of a circuit diagram.

    This maintains an ordered list of qubits, and for each qubit a 'line': that is, a list of
    TikZ operators.
    """

    def __init__(self, qubits: Sequence[int]):
        self.qubits = qubits
        self.lines: Mapping[int, List[str]] = defaultdict(list)

    def extend_lines_to_common_edge(self, qubits: Iterable[int], offset: int=0) -> None:
        """
        Add NOP operations on the lines associated with the given qubits, until
        all lines are of the same width.
        """
        max_width = max((self.width(q) for q in qubits)) + offset
        for q in qubits:
            while self.width(q) < max_width:
                self.append(q, TIKZ_NOP())

    def width(self, qubit: int) -> int:
        """
        The width of the diagram, in terms of the number of operations, on the
        specified qubit line.
        """
        return len(self.lines[qubit])

    def append(self, qubit: int, op: str) -> None:
        """
        Add an operation to the rightmost edge of the specified qubit line.
        """
        self.lines[qubit].append(op)

    def append_diagram(self, diagram: 'DiagramState', group: Optional[str]=None) -> 'DiagramState':
        """
        Add all operations represented by the given diagram to their
        corresponding qubit lines in this diagram.

        If group is not None, then a TIKZ_GATE_GROUP is created with the label indicated by group.
        """
        grouped_qubits = diagram.qubits
        diagram.extend_lines_to_common_edge(grouped_qubits)
        self.extend_lines_to_common_edge(grouped_qubits)
        corner_row = grouped_qubits[0]
        corner_col = len(self.lines[corner_row]) + 1
        group_width = diagram.width(corner_row) - 1
        for q in diagram.qubits:
            for op in diagram.lines[q]:
                self.append(q, op)
        if group is not None:
            self.lines[corner_row][corner_col] += ' ' + TIKZ_GATE_GROUP(grouped_qubits, group_width, group)
        return self

    def interval(self, low: int, high: int) -> List[int]:
        """
        All qubits in the diagram, from low to high, inclusive.
        """
        full_interval = range(low, high + 1)
        qubits = list(set(full_interval) & set(self.qubits))
        return sorted(qubits)

    def is_interval(self, qubits: Sequence[int]) -> bool:
        """
        Do the specified qubits correspond to an interval in this diagram?
        """
        return qubits == self.interval(min(qubits), max(qubits))