from typing import TYPE_CHECKING, List, Tuple, cast, Dict
import matplotlib.textpath
import matplotlib.font_manager
class SVGCircuit:
    """A wrapper around cirq.Circuit to enable rich display in a Jupyter
    notebook.

    Jupyter will display the result of the last line in a cell. Often,
    this is repr(o) for an object. This class defines a magic method
    which will cause the circuit to be displayed as an SVG image.
    """

    def __init__(self, circuit: 'cirq.Circuit'):
        self.circuit = circuit

    def _repr_svg_(self) -> str:
        return circuit_to_svg(self.circuit)