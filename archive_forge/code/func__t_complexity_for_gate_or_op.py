from typing import Any, Callable, Hashable, Iterable, Optional, Union, overload
import attr
import cachetools
import cirq
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
from typing_extensions import Literal, Protocol
from cirq_ft.deprecation import deprecated_cirq_ft_class, deprecated_cirq_ft_function
@cachetools.cached(cachetools.LRUCache(128), key=_get_hash, info=True)
def _t_complexity_for_gate_or_op(gate_or_op: Union[cirq.Gate, cirq.Operation], fail_quietly: bool) -> Optional[TComplexity]:
    strategies = [_has_t_complexity, _is_clifford_or_t, _from_decomposition]
    return _t_complexity_from_strategies(gate_or_op, fail_quietly, strategies)