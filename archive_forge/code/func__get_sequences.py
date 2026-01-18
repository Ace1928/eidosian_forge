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
def _get_sequences(self, instruction: PulseQobjInstruction) -> Iterator[instructions.Instruction]:
    """A method to iterate over pulse instructions without creating Schedule.

        .. note::

            This is internal fast-path function, and callers other than this converter class
            might directly use this method to generate schedule from multiple
            Qobj instructions. Because __call__ always returns a schedule with the time offset
            parsed instruction, composing multiple Qobj instructions to create
            a gate schedule is somewhat inefficient due to composing overhead of schedules.
            Directly combining instructions with this method is much performant.

        Args:
            instruction: Instruction data in Qobj format.

        Yields:
            Qiskit Pulse instructions.

        :meta public:
        """
    try:
        method = getattr(self, f'_convert_{instruction.name}')
    except AttributeError:
        method = self._convert_generic
    yield from method(instruction)