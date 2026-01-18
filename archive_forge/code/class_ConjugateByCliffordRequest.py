import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class ConjugateByCliffordRequest(Message):
    """
    RPC request payload for conjugating a Pauli element by a Clifford element.
    """
    pauli: PauliTerm
    'Specification of a Pauli element.'
    clifford: str
    'Specification of a Clifford element.'