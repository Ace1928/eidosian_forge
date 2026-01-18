import itertools
from qiskit.circuit.exceptions import CircuitError
from .register import Register
from .bit import Bit
class AncillaQubit(Qubit):
    """A qubit used as ancillary qubit."""
    __slots__ = ()
    pass