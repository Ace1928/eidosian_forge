from __future__ import annotations
import typing
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Type
import numpy as np
from qiskit.pulse import channels as chans, exceptions, instructions
from qiskit.pulse.channels import ClassicalIOChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.exceptions import UnassignedDurationError
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock, ScheduleComponent
def _inline_block(block: ScheduleBlock) -> ScheduleBlock:
    """A helper function to inline subroutine of schedule block.

    .. note:: If subroutine is ``Schedule`` the function raises an error.
    """
    ret_block = ScheduleBlock.initialize_from(block)
    for inst in block.blocks:
        if isinstance(inst, ScheduleBlock):
            inline_block = _inline_block(inst)
            ret_block.append(inline_block, inplace=True)
        else:
            ret_block.append(inst, inplace=True)
    return ret_block