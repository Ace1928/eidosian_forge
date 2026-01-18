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
def _convert_shiftf(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
    """Return converted `ShiftFrequency` instruction.

        .. note::

            We assume frequency value is expressed in string with "GHz".
            Operand value is thus scaled by a factor of 1e9.

        Args:
            instruction: ShiftFrequency qobj instruction

        Yields:
            Qiskit Pulse shift frequency schedule instructions
        """
    channel = self.get_channel(instruction.ch)
    frequency = self.disassemble_value(instruction.frequency) * 1000000000.0
    yield instructions.ShiftFrequency(frequency, channel)