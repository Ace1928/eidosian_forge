import copy
from itertools import product
from typing import Callable, List, Sequence, Tuple, Union
from networkx import MultiDiGraph
import pennylane as qml
from pennylane import expval
from pennylane.measurements import ExpectationMP, MeasurementProcess, SampleMP
from pennylane.operation import Operator, Tensor
from pennylane.ops.meta import WireCut
from pennylane.pauli import string_to_pauli_word
from pennylane.queuing import WrappedObj
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.wires import Wires
from .utils import MeasureNode, PrepareNode
def _find_new_wire(wires: Wires) -> int:
    """Finds a new wire label that is not in ``wires``."""
    ctr = 0
    while ctr in wires:
        ctr += 1
    return ctr