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
def _cut_circuit_expand(tape: QuantumTape, use_opt_einsum: bool=False, device_wires: Optional[Wires]=None, max_depth: int=1, auto_cutter: Union[bool, Callable]=False, **kwargs) -> (Sequence[QuantumTape], Callable):
    """Main entry point for expanding operations until reaching a depth that
    includes :class:`~.WireCut` operations."""

    def processing_fn(res):
        return res[0]
    tapes, tapes_fn = ([tape], processing_fn)
    tape_meas_ops = tape.measurements
    if tape_meas_ops and isinstance(tape_meas_ops[0].obs, qml.Hamiltonian):
        if len(tape_meas_ops) > 1:
            raise NotImplementedError('Hamiltonian expansion is supported only with a single Hamiltonian')
        new_meas_op = type(tape_meas_ops[0])(obs=qml.Hamiltonian(*tape_meas_ops[0].obs.terms()))
        new_tape = type(tape)(tape.operations, [new_meas_op], shots=tape.shots, trainable_params=tape.trainable_params)
        tapes, tapes_fn = qml.transforms.hamiltonian_expand(new_tape, group=False)
    return ([_qcut_expand_fn(tape, max_depth, auto_cutter) for tape in tapes], tapes_fn)