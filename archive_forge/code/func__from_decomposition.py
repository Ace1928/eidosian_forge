from typing import Any, Callable, Hashable, Iterable, Optional, Union, overload
import attr
import cachetools
import cirq
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
from typing_extensions import Literal, Protocol
from cirq_ft.deprecation import deprecated_cirq_ft_class, deprecated_cirq_ft_function
def _from_decomposition(stc: Any, fail_quietly: bool) -> Optional[TComplexity]:
    decomposition = _decompose_once_considering_known_decomposition(stc)
    if decomposition is None:
        return None
    return _is_iterable(decomposition, fail_quietly=fail_quietly)