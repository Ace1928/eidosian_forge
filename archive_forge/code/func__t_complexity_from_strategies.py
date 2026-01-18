from typing import Any, Callable, Hashable, Iterable, Optional, Union, overload
import attr
import cachetools
import cirq
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
from typing_extensions import Literal, Protocol
from cirq_ft.deprecation import deprecated_cirq_ft_class, deprecated_cirq_ft_function
def _t_complexity_from_strategies(stc: Any, fail_quietly: bool, strategies: Iterable[Callable[[Any, bool], Optional[TComplexity]]]):
    ret = None
    for strategy in strategies:
        ret = strategy(stc, fail_quietly)
        if ret is not None:
            break
    return ret