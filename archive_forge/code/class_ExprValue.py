from abc import abstractmethod
from enum import Enum, IntEnum
import numpy as np
from qiskit.circuit import (
from qiskit.circuit.annotated_operation import AnnotatedOperation, Modifier
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.pulse.channels import (
from qiskit.pulse.configuration import Discriminator, Kernel
from qiskit.pulse.instructions import (
from qiskit.pulse.library import Waveform, SymbolicPulse
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.pulse.transforms.alignments import (
from qiskit.qpy import exceptions
class ExprValue(TypeKeyBase):
    """Type keys for the ``EXPR_VALUE`` QPY item."""
    BOOL = b'b'
    INT = b'i'

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, bool):
            return cls.BOOL
        if isinstance(obj, int):
            return cls.INT
        raise exceptions.QpyError(f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace.")

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError