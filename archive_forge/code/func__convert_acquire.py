import hashlib
import re
import warnings
from enum import Enum
from functools import singledispatchmethod
from typing import Union, List, Iterator, Optional
import numpy as np
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse import channels, instructions, library
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import QiskitError
from qiskit.pulse.parser import parse_string_expr
from qiskit.pulse.schedule import Schedule
from qiskit.qobj import QobjMeasurementOption, PulseLibraryItem, PulseQobjInstruction
from qiskit.qobj.utils import MeasLevel
def _convert_acquire(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
    """Return converted `Acquire` instruction.

        Args:
            instruction: Acquire qobj

        Yields:
            Qiskit Pulse acquire instructions
        """
    duration = instruction.duration
    qubits = instruction.qubits
    acquire_channels = [channels.AcquireChannel(qubit) for qubit in qubits]
    mem_slots = [channels.MemorySlot(instruction.memory_slot[i]) for i in range(len(qubits))]
    if hasattr(instruction, 'register_slot'):
        register_slots = [channels.RegisterSlot(instruction.register_slot[i]) for i in range(len(qubits))]
    else:
        register_slots = [None] * len(qubits)
    discriminators = instruction.discriminators if hasattr(instruction, 'discriminators') else None
    if not isinstance(discriminators, list):
        discriminators = [discriminators]
    if any((discriminators[i] != discriminators[0] for i in range(len(discriminators)))):
        warnings.warn('Can currently only support one discriminator per acquire. Defaulting to first discriminator entry.')
    discriminator = discriminators[0]
    if discriminator:
        discriminator = Discriminator(name=discriminators[0].name, **discriminators[0].params)
    kernels = instruction.kernels if hasattr(instruction, 'kernels') else None
    if not isinstance(kernels, list):
        kernels = [kernels]
    if any((kernels[0] != kernels[i] for i in range(len(kernels)))):
        warnings.warn('Can currently only support one kernel per acquire. Defaulting to first kernel entry.')
    kernel = kernels[0]
    if kernel:
        kernel = Kernel(name=kernels[0].name, **kernels[0].params)
    for acquire_channel, mem_slot, reg_slot in zip(acquire_channels, mem_slots, register_slots):
        yield instructions.Acquire(duration, acquire_channel, mem_slot=mem_slot, reg_slot=reg_slot, kernel=kernel, discriminator=discriminator)