import functools
import itertools
import math
import operator
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, protocols, value
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.permutation import (
def acquaint_and_shift(parts: Tuple[List['cirq.Qid'], List['cirq.Qid']], layers: Layers, acquaintance_size: Optional[int], swap_gate: 'cirq.Gate', mapping: Dict[ops.Qid, int]):
    """Acquaints and shifts a pair of lists of qubits. The first part is
    acquainted with every qubit individually in the second part, and vice
    versa. Operations are grouped into several layers:
        * prior_interstitial: The first layer of acquaintance gates.
        * prior: The combination of acquaintance gates and swaps that acquaints
            the inner halves.
        * intra: The shift gate.
        * post: The combination of acquaintance gates and swaps that acquaints
            the outer halves.
        * posterior_interstitial: The last layer of acquaintance gates.

    Args:
        parts: The two lists of qubits to acquaint.
        layers: The layers to put gates into.
        acquaintance_size: The number of qubits to acquaint at a time. If None,
            after each pair of parts is shifted the union thereof is
            acquainted.
        swap_gate: The gate used to swap logical indices.
        mapping: The mapping from qubits to logical indices. Used to keep track
            of the effect of inside-acquainting swaps.
    """
    left_part, right_part = parts
    left_size, right_size = (len(left_part), len(right_part))
    assert not set(left_part) & set(right_part)
    qubits = left_part + right_part
    shift = CircularShiftGate(len(qubits), left_size, swap_gate=swap_gate)(*qubits)
    if acquaintance_size is None:
        layers.intra.append(shift)
        layers.post.append(acquaint(*qubits))
        shift.gate.update_mapping(mapping, qubits)
    elif max(left_size, right_size) != acquaintance_size - 1:
        layers.intra.append(shift)
        shift.gate.update_mapping(mapping, qubits)
    elif acquaintance_size == 2:
        layers.prior_interstitial.append(acquaint(*qubits))
        layers.intra.append(shift)
        shift.gate.update_mapping(mapping, qubits)
    else:
        if left_size == acquaintance_size - 1:
            pre_acquaintance_gate = acquaint(*qubits[:acquaintance_size])
            acquaint_insides(swap_gate=swap_gate, acquaintance_gate=pre_acquaintance_gate, qubits=right_part, before=True, layers=layers, mapping=mapping)
        if right_size == acquaintance_size - 1:
            pre_acquaintance_gate = acquaint(*qubits[-acquaintance_size:])
            acquaint_insides(swap_gate=swap_gate, acquaintance_gate=pre_acquaintance_gate, qubits=left_part[::-1], before=True, layers=layers, mapping=mapping)
        layers.intra.append(shift)
        shift.gate.update_mapping(mapping, qubits)
        if left_size == acquaintance_size - 1 and right_size > 1:
            post_acquaintance_gate = acquaint(*qubits[-acquaintance_size:])
            new_left_part = qubits[right_size - 1::-1]
            acquaint_insides(swap_gate=swap_gate, acquaintance_gate=post_acquaintance_gate, qubits=new_left_part, before=False, layers=layers, mapping=mapping)
        if right_size == acquaintance_size - 1 and left_size > 1:
            post_acquaintance_gate = acquaint(*qubits[:acquaintance_size])
            acquaint_insides(swap_gate=swap_gate, acquaintance_gate=post_acquaintance_gate, qubits=qubits[right_size:], before=False, layers=layers, mapping=mapping)