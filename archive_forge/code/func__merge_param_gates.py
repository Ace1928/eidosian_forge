import math
import warnings
from itertools import product
from typing import Sequence, Callable
import pennylane as qml
from pennylane.ops import Adjoint
from pennylane.queuing import QueuingManager
from pennylane.transforms.core import transform
from pennylane.tape import QuantumTape
from pennylane.transforms.optimization import (
from pennylane.transforms.optimization.optimization_utils import find_next_gate, _fuse_global_phases
from pennylane.ops.op_math.decompositions.solovay_kitaev import sk_decomposition
def _merge_param_gates(operations, merge_ops=None):
    """Merge the provided parameterized gates on the same wires that are adjacent to each other"""
    copied_ops = operations.copy()
    merged_ops, number_ops = ([], 0)
    while len(copied_ops) > 0:
        curr_gate = copied_ops.pop(0)
        if merge_ops is not None and curr_gate.name not in merge_ops:
            merged_ops.append(curr_gate)
            continue
        if curr_gate.name in merge_ops:
            number_ops += 1
        next_gate_idx = find_next_gate(curr_gate.wires, copied_ops)
        if next_gate_idx is None:
            merged_ops.append(curr_gate)
            continue
        curr_params = curr_gate.parameters
        curr_intrfc = qml.math.get_deep_interface(curr_gate.parameters)
        cumulative_angles = qml.math.array(curr_params, dtype=float, like=curr_intrfc)
        next_gate = copied_ops[next_gate_idx]
        while curr_gate.name == next_gate.name and curr_gate.wires == next_gate.wires:
            cumulative_angles += qml.math.array(next_gate.parameters, like=curr_intrfc)
            copied_ops.pop(next_gate_idx)
            next_gate_idx = find_next_gate(curr_gate.wires, copied_ops)
            if next_gate_idx is None:
                break
            next_gate = copied_ops[next_gate_idx]
        merged_ops.append(curr_gate.__class__(*cumulative_angles, wires=curr_gate.wires))
    return (merged_ops, number_ops)