import copy
import logging
import uuid
import warnings
from time import time
from typing import Dict, List, Optional, Union
import numpy as np
from qiskit.assembler import assemble_circuits, assemble_schedules
from qiskit.assembler.run_config import RunConfig
from qiskit.circuit import Parameter, QuantumCircuit, Qubit
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.pulse import Instruction, LoConfig, Schedule, ScheduleBlock
from qiskit.pulse.channels import PulseChannel
from qiskit.qobj import QasmQobj, PulseQobj, QobjHeader
from qiskit.qobj.utils import MeasLevel, MeasReturnType
def _parse_rep_delay(rep_delay: float, default_rep_delay: float, rep_delay_range: List[float]) -> float:
    """Parse and set ``rep_delay`` parameter in runtime config.

    Args:
        rep_delay: Initial rep delay.
        default_rep_delay: Backend default rep delay.
        rep_delay_range: Backend list defining allowable range of rep delays.

    Raises:
        QiskitError: If rep_delay is not in the backend rep_delay_range.
    Returns:
        float: Modified rep delay after parsing.
    """
    if rep_delay is None:
        rep_delay = default_rep_delay
    if rep_delay is not None:
        if rep_delay_range is not None and isinstance(rep_delay_range, list):
            if len(rep_delay_range) != 2:
                raise QiskitError('Backend rep_delay_range {} must be a list with two entries.'.format(rep_delay_range))
            if not rep_delay_range[0] <= rep_delay <= rep_delay_range[1]:
                raise QiskitError('Supplied rep delay {} not in the supported backend range {}'.format(rep_delay, rep_delay_range))
        rep_delay = rep_delay * 1000000.0
    return rep_delay