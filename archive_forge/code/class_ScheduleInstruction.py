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
class ScheduleInstruction(TypeKeyBase):
    """Type key enum for schedule instruction object."""
    ACQUIRE = b'a'
    PLAY = b'p'
    DELAY = b'd'
    SET_FREQUENCY = b'f'
    SHIFT_FREQUENCY = b'g'
    SET_PHASE = b'q'
    SHIFT_PHASE = b'r'
    BARRIER = b'b'
    TIME_BLOCKADE = b't'
    REFERENCE = b'y'

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, Acquire):
            return cls.ACQUIRE
        if isinstance(obj, Play):
            return cls.PLAY
        if isinstance(obj, Delay):
            return cls.DELAY
        if isinstance(obj, SetFrequency):
            return cls.SET_FREQUENCY
        if isinstance(obj, ShiftFrequency):
            return cls.SHIFT_FREQUENCY
        if isinstance(obj, SetPhase):
            return cls.SET_PHASE
        if isinstance(obj, ShiftPhase):
            return cls.SHIFT_PHASE
        if isinstance(obj, RelativeBarrier):
            return cls.BARRIER
        if isinstance(obj, TimeBlockade):
            return cls.TIME_BLOCKADE
        if isinstance(obj, Reference):
            return cls.REFERENCE
        raise exceptions.QpyError(f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace.")

    @classmethod
    def retrieve(cls, type_key):
        if type_key == cls.ACQUIRE:
            return Acquire
        if type_key == cls.PLAY:
            return Play
        if type_key == cls.DELAY:
            return Delay
        if type_key == cls.SET_FREQUENCY:
            return SetFrequency
        if type_key == cls.SHIFT_FREQUENCY:
            return ShiftFrequency
        if type_key == cls.SET_PHASE:
            return SetPhase
        if type_key == cls.SHIFT_PHASE:
            return ShiftPhase
        if type_key == cls.BARRIER:
            return RelativeBarrier
        if type_key == cls.TIME_BLOCKADE:
            return TimeBlockade
        if type_key == cls.REFERENCE:
            return Reference
        raise exceptions.QpyError(f"A class corresponding to type key '{type_key}' is not found in {cls.__name__} namespace.")