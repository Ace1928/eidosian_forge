import abc
from typing import Callable, Dict, Iterator, List, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_function
def _unary_iteration_segtree(ops: List[cirq.Operation], control: cirq.Qid, selection: Sequence[cirq.Qid], ancilla: Sequence[cirq.Qid], sl: int, l: int, r: int, l_iter: int, r_iter: int, break_early: Callable[[int, int], bool]) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    """Constructs a unary iteration circuit by iterating over nodes of an implicit Segment Tree.

    Args:
        ops: Operations accumulated so far while traversing the implicit segment tree. The
            accumulated ops are yielded and cleared when we reach a leaf node.
        control: The control qubit that controls the execution of the entire unary iteration
            circuit represented by the current node of the segment tree.
        selection: Sequence of selection qubits. The i'th qubit in the list corresponds to the i'th
            level in the segment tree.Thus, a total of O(logN) selection qubits are required for a
            tree on range `N = (r_iter - l_iter)`.
        ancilla: Pre-allocated ancilla qubits to be used for constructing the unary iteration
            circuit.
        sl: Current depth of the tree. `selection[sl]` gives the selection qubit corresponding to
            the current depth.
        l: Left index of the range represented by current node of the segment tree.
        r: Right index of the range represented by current node of the segment tree.
        l_iter: Left index of iteration range over which the segment tree should be constructed.
        r_iter: Right index of iteration range over which the segment tree should be constructed.
        break_early: For each internal node of the segment tree, `break_early(l, r)` is called to
            evaluate whether the unary iteration should terminate early and not recurse in the
            subtree of the node representing range `[l, r)`. If True, the internal node is
            considered equivalent to a leaf node and the method yields only one tuple
            `(OP_TREE, control_qubit, l)` for all integers in the range `[l, r)`.

    Yields:
        One `Tuple[cirq.OP_TREE, cirq.Qid, int]` for each leaf node in the segment tree. The i'th
        yielded element corresponds to the i'th leaf node which represents the `l_iter + i`'th
        integer. The tuple corresponds to:
            - cirq.OP_TREE: Operations to be inserted in the circuit in between the last leaf node
                (or beginning of iteration) to the current leaf node.
            - cirq.Qid: The control qubit which can be controlled upon to execute the $U_{l}$ on a
                target register when the selection register stores integer $l$.
            - int: Integer $l$ which would be stored in the selection register if the control qubit
                 is set.
    """
    if l >= r_iter or l_iter >= r:
        return
    if l_iter <= l < r <= r_iter and (l == r - 1 or break_early(l, r)):
        yield (tuple(ops), control, l)
        ops.clear()
        return
    assert sl < len(selection)
    m = l + r >> 1
    if r_iter <= m:
        yield from _unary_iteration_segtree(ops, control, selection, ancilla, sl + 1, l, m, l_iter, r_iter, break_early)
        return
    if l_iter >= m:
        yield from _unary_iteration_segtree(ops, control, selection, ancilla, sl + 1, m, r, l_iter, r_iter, break_early)
        return
    anc, sq = (ancilla[sl], selection[sl])
    ops.append(and_gate.And((1, 0)).on(control, sq, anc))
    yield from _unary_iteration_segtree(ops, anc, selection, ancilla, sl + 1, l, m, l_iter, r_iter, break_early)
    ops.append(cirq.CNOT(control, anc))
    yield from _unary_iteration_segtree(ops, anc, selection, ancilla, sl + 1, m, r, l_iter, r_iter, break_early)
    ops.append(and_gate.And(adjoint=True).on(control, sq, anc))