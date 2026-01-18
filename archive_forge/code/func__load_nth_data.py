from typing import Callable, Sequence, Tuple
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate, unary_iteration_gate
from numpy.typing import ArrayLike, NDArray
def _load_nth_data(self, selection_idx: Tuple[int, ...], gate: Callable[[cirq.Qid], cirq.Operation], **target_regs: NDArray[cirq.Qid]) -> cirq.OP_TREE:
    for i, d in enumerate(self.data):
        target = target_regs.get(f'target{i}', ())
        for q, bit in zip(target, f'{int(d[selection_idx]):0{len(target)}b}'):
            if int(bit):
                yield gate(q)