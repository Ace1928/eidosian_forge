from typing import (
from cirq import protocols, value
from cirq.ops import (
def _op_repr_(self, qubits: Sequence['cirq.Qid']) -> str:
    args = [repr(self._observable.on(*qubits))]
    if self.key != _default_measurement_key(qubits):
        args.append(f'key={self.mkey!r}')
    arg_list = ', '.join(args)
    return f'cirq.measure_single_paulistring({arg_list})'