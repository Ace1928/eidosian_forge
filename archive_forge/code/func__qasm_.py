import string
from typing import TYPE_CHECKING, Union, Any, Tuple, TypeVar, Optional, Dict, Iterable
from typing_extensions import Protocol
from cirq import ops
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
@doc_private
def _qasm_(self, qubits: Tuple['cirq.Qid'], args: QasmArgs) -> Union[None, NotImplementedType, str]:
    pass