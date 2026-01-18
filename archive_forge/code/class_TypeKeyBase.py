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
class TypeKeyBase(bytes, Enum):
    """Abstract baseclass for type key Enums."""

    @classmethod
    @abstractmethod
    def assign(cls, obj):
        """Assign type key to given object.

        Args:
            obj (any): Arbitrary object to evaluate.

        Returns:
            TypeKey: Corresponding key object.
        """
        pass

    @classmethod
    @abstractmethod
    def retrieve(cls, type_key):
        """Get a class from given type key.

        Args:
            type_key (bytes): Object type key.

        Returns:
            any: Corresponding class.
        """
        pass