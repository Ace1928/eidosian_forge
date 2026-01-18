import string
from typing import TYPE_CHECKING, Union, Any, Tuple, TypeVar, Optional, Dict, Iterable
from typing_extensions import Protocol
from cirq import ops
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
class SupportsQasm(Protocol):
    """An object that can be turned into QASM code.

    Returning `NotImplemented` or `None` means "don't know how to turn into
    QASM". In that case fallbacks based on decomposition and known unitaries
    will be used instead.
    """

    @doc_private
    def _qasm_(self) -> Union[None, NotImplementedType, str]:
        pass