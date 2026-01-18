from typing import Tuple
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate
class MultiTargetCNOT(infra.GateWithRegisters):
    """Implements single control, multi-target CNOT_{n} gate in 2*log(n) + 1 CNOT depth.

    Implements CNOT_{n} = |0><0| I + |1><1| X^{n} using a circuit of depth 2*log(n) + 1
    containing only CNOT gates. See Appendix B.1 of https://arxiv.org/abs/1812.00954 for
    reference.
    """

    def __init__(self, num_targets: int):
        self._num_targets = num_targets

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature.build(control=1, targets=self._num_targets)

    def decompose_from_registers(self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]):
        control, targets = (quregs['control'], quregs['targets'])

        def cnots_for_depth_i(i: int, q: NDArray[cirq.Qid]) -> cirq.OP_TREE:
            for c, t in zip(q[:2 ** i], q[2 ** i:min(len(q), 2 ** (i + 1))]):
                yield cirq.CNOT(c, t)
        depth = len(targets).bit_length()
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(depth - i - 1, targets))
        yield cirq.CNOT(*control, targets[0])
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(i, targets))

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=['@'] + ['X'] * self._num_targets)