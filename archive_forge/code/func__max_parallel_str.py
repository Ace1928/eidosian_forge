from typing import Any, cast, Optional, Type, Union
from cirq.ops import gateset, raw_types, parallel_gate, eigen_gate
from cirq import protocols
def _max_parallel_str(self):
    return self._max_parallel_allowed if self._max_parallel_allowed is not None else 'INF'