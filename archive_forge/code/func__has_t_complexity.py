from typing import Any, Callable, Hashable, Iterable, Optional, Union, overload
import attr
import cachetools
import cirq
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
from typing_extensions import Literal, Protocol
from cirq_ft.deprecation import deprecated_cirq_ft_class, deprecated_cirq_ft_function
def _has_t_complexity(stc: Any, fail_quietly: bool) -> Optional[TComplexity]:
    """Returns TComplexity of stc by calling `stc._t_complexity_()` method, if it exists."""
    estimator = getattr(stc, '_t_complexity_', None)
    if estimator is not None:
        result = estimator()
        if result is not NotImplemented:
            return result
    if isinstance(stc, cirq.Operation) and stc.gate is not None:
        return _has_t_complexity(stc.gate, fail_quietly)
    return None