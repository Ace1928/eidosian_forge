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
class ExprType(TypeKeyBase):
    """Type keys for the ``EXPR_TYPE`` QPY item."""
    BOOL = b'b'
    UINT = b'u'

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, types.Type) and (key := getattr(cls, obj.__class__.__name__.upper(), None)) is not None:
            return key
        raise exceptions.QpyError(f"Object '{obj}' is not supported in {cls.__name__} namespace.")

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError