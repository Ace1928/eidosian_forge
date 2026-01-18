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
def _convert_generic(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
    """Convert generic pulse instruction.

        Args:
            instruction: Generic qobj instruction

        Yields:
            Qiskit Pulse generic instructions

        Raises:
            QiskitError: When instruction name not found.
        """
    if instruction.name in self._pulse_library:
        waveform = library.Waveform(samples=self._pulse_library[instruction.name], name=instruction.name)
        channel = self.get_channel(instruction.ch)
        yield instructions.Play(waveform, channel)
    else:
        if (qubits := getattr(instruction, 'qubits', None)):
            msg = f'qubits {qubits}'
        else:
            msg = f'channel {instruction.ch}'
        raise QiskitError(f'Instruction {instruction.name} on {msg} is not found in Qiskit namespace. This instruction cannot be deserialized.')