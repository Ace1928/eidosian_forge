import re
from fractions import Fraction
from typing import (
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq import protocols, value
from cirq._doc import doc_private
class SupportsCircuitDiagramInfo(Protocol):
    """A diagrammable operation on qubits."""

    @doc_private
    def _circuit_diagram_info_(self, args: CircuitDiagramInfoArgs) -> Union[str, Iterable[str], CircuitDiagramInfo]:
        """Describes how to draw an operation in a circuit diagram.

        This method is used by the global `cirq.diagram_info` method. If this
        method is not present, or returns NotImplemented, it is assumed that the
        receiving object doesn't specify diagram info.

        Args:
            args: A DiagramInfoArgs instance encapsulating various pieces of
                information (e.g. how many qubits are we being applied to) as
                well as user options (e.g. whether to avoid unicode characters).

        Returns:
            A DiagramInfo instance describing what to show.
        """