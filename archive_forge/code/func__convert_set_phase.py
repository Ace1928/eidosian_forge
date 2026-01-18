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
@_convert_instruction.register(instructions.SetPhase)
def _convert_set_phase(self, instruction, time_offset: int) -> PulseQobjInstruction:
    """Return converted `SetPhase`.

        Args:
            instruction: Qiskit Pulse set phase instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
    command_dict = {'name': 'setp', 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'phase': instruction.phase}
    return self._qobj_model(**command_dict)