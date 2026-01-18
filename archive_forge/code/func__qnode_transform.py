from functools import partial
from typing import Callable, Optional, Union, Sequence
import pennylane as qml
from pennylane.measurements import ExpectationMP
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.wires import Wires
from .cutstrategy import CutStrategy
from .kahypar import kahypar_cut
from .processing import qcut_processing_fn
from .tapes import _qcut_expand_fn, expand_fragment_tape, graph_to_tape, tape_to_graph
from .utils import find_and_place_cuts, fragment_graph, replace_wire_cut_nodes
@cut_circuit.custom_qnode_transform
def _qnode_transform(self, qnode, targs, tkwargs):
    """Here, we overwrite the QNode execution wrapper in order
    to access the device wires."""
    tkwargs.setdefault('device_wires', qnode.device.wires)
    return self.default_qnode_transform(qnode, targs, tkwargs)