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
def _convert_delay(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
    """Return converted `Delay` instruction.

        Args:
            instruction: Delay qobj instruction

        Yields:
            Qiskit Pulse delay instructions
        """
    channel = self.get_channel(instruction.ch)
    duration = instruction.duration
    yield instructions.Delay(duration, channel)