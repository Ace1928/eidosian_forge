from typing import Tuple
from numpy.typing import NDArray
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq._compat import cached_property
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
class CustomProgrammableRotationGateArray(cirq_ft.ProgrammableRotationGateArrayBase):

    def interleaved_unitary(self, index: int, **qubit_regs: NDArray[cirq.Qid]) -> cirq.Operation:
        two_qubit_ops_factory = [cirq.X(*qubit_regs['unrelated_target']).controlled_by(*qubit_regs['rotations_target']), cirq.Z(*qubit_regs['unrelated_target']).controlled_by(*qubit_regs['rotations_target'])]
        return two_qubit_ops_factory[index % 2]

    @cached_property
    def interleaved_unitary_target(self) -> Tuple[cirq_ft.Register, ...]:
        return tuple(cirq_ft.Signature.build(unrelated_target=1))