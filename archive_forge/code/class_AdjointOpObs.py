from functools import wraps
import pennylane as qml
from pennylane.math import conj, moveaxis, transpose
from pennylane.operation import Observable, Operation, Operator
from pennylane.queuing import QueuingManager
from pennylane.tape import make_qscript
from pennylane.compiler import compiler
from pennylane.compiler.compiler import CompileError
from .symbolicop import SymbolicOp
class AdjointOpObs(AdjointOperation, Observable):
    """A child of :class:`~.AdjointOperation` that also inherits from :class:`~.Observable."""

    def __new__(cls, *_, **__):
        return object.__new__(cls)