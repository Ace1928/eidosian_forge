import dataclasses
import math
from typing import Iterable, Callable
from qiskit.circuit import (
from qiskit._qasm2 import (  # pylint: disable=no-name-in-module
from .exceptions import QASM2ParseError
@dataclasses.dataclass(frozen=True)
class CustomInstruction:
    """Information about a custom instruction that should be defined during the parse.

    The ``name``, ``num_params`` and ``num_qubits`` fields are self-explanatory.  The
    ``constructor`` field should be a callable object with signature ``*args -> Instruction``, where
    each of the ``num_params`` ``args`` is a floating-point value.  Most of the built-in Qiskit gate
    classes have this form.

    There is a final ``builtin`` field.  This is optional, and if set true will cause the
    instruction to be defined and available within the parsing, even if there is no definition in
    any included OpenQASM 2 file.
    """
    name: str
    num_params: int
    num_qubits: int
    constructor: Callable[..., Instruction]
    builtin: bool = False