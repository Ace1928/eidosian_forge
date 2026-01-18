from typing import Optional, Callable, Hashable, Sequence, TYPE_CHECKING
from cirq import circuits
from cirq.protocols import decompose_protocol as dp
from cirq.transformers import transformer_api, transformer_primitives
def _create_on_stuck_raise_error(gateset: 'cirq.Gateset'):

    def _value_error_describing_bad_operation(op: 'cirq.Operation') -> ValueError:
        return ValueError(f'Unable to convert {op} to target gateset {gateset!r}')
    return _value_error_describing_bad_operation