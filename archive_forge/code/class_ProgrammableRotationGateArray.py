import abc
from typing import Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_method, cached_property
from cirq_ft import infra
from cirq_ft.algos import qrom
from cirq_ft.infra.bit_tools import iter_bits
class ProgrammableRotationGateArray(ProgrammableRotationGateArrayBase):
    """An implementation of `ProgrammableRotationGateArrayBase` base class


    This implementation of the `ProgrammableRotationGateArray` base class expects
    all interleaved_unitaries to act on the `rotations_target` register.

    See docstring of `ProgrammableRotationGateArrayBase` for more details.
    """

    def __init__(self, *angles: Sequence[int], kappa: int, rotation_gate: cirq.Gate, interleaved_unitaries: Sequence[cirq.Gate]=()):
        super().__init__(*angles, kappa=kappa, rotation_gate=rotation_gate)
        if not interleaved_unitaries:
            identity_gate = cirq.IdentityGate(infra.total_bits(self.rotations_target))
            interleaved_unitaries = (identity_gate,) * (len(angles) - 1)
        assert len(interleaved_unitaries) == len(angles) - 1
        assert all((cirq.num_qubits(u) == self._target_bitsize for u in interleaved_unitaries))
        self._interleaved_unitaries = tuple(interleaved_unitaries)

    def interleaved_unitary(self, index: int, **qubit_regs: NDArray[cirq.Qid]) -> cirq.Operation:
        return self._interleaved_unitaries[index].on(*qubit_regs['rotations_target'])

    @cached_property
    def interleaved_unitary_target(self) -> Tuple[infra.Register, ...]:
        return ()