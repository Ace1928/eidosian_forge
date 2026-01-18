from __future__ import annotations
import enum
import warnings
from collections.abc import Sequence
from math import pi, erf
import numpy as np
from qiskit.circuit import Instruction as CircuitInst
from qiskit.circuit.library.standard_gates import RZXGate
from qiskit.exceptions import QiskitError
from qiskit.pulse import (
from qiskit.pulse import builder
from qiskit.pulse.filters import filter_instructions
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.transpiler.target import Target
from .base_builder import CalibrationBuilder
from .exceptions import CalibrationNotAvailable
def _check_calibration_type(inst_sched_map: InstructionScheduleMap, qubits: Sequence[int]) -> tuple[CRCalType, list[Play], list[Play]]:
    """A helper function to check type of CR calibration.

    Args:
        inst_sched_map: instruction schedule map of the backends
        qubits: ordered tuple of qubits for cross resonance (q_control, q_target)

    Returns:
        Filtered instructions and most-likely type of calibration.

    Raises:
        QiskitError: Unknown calibration type is detected.
    """
    cal_type = None
    if inst_sched_map.has('cx', qubits):
        cr_sched = inst_sched_map.get('cx', qubits=qubits)
    elif inst_sched_map.has('ecr', qubits):
        cr_sched = inst_sched_map.get('ecr', qubits=qubits)
        cal_type = CRCalType.ECR_FORWARD
    elif inst_sched_map.has('ecr', tuple(reversed(qubits))):
        cr_sched = inst_sched_map.get('ecr', tuple(reversed(qubits)))
        cal_type = CRCalType.ECR_REVERSE
    else:
        raise QiskitError(f'Native direction cannot be determined: operation on qubits {qubits} for the following instruction schedule map:\n{inst_sched_map}')
    cr_tones = [t[1] for t in filter_instructions(cr_sched, [_filter_cr_tone]).instructions]
    comp_tones = [t[1] for t in filter_instructions(cr_sched, [_filter_comp_tone]).instructions]
    if cal_type is None:
        if len(comp_tones) == 0:
            raise QiskitError(f'{repr(cr_sched)} has no target compensation tones. Native ECR direction cannot be determined.')
        if comp_tones[0].channel.index == qubits[1]:
            cal_type = CRCalType.ECR_CX_FORWARD
        else:
            cal_type = CRCalType.ECR_CX_REVERSE
    if len(cr_tones) == 2 and len(comp_tones) in (0, 2):
        return (cal_type, cr_tones, comp_tones)
    if len(cr_tones) == 1 and len(comp_tones) == 1:
        if comp_tones[0].channel.index == qubits[1]:
            return (CRCalType.DIRECT_CX_FORWARD, cr_tones, comp_tones)
        else:
            return (CRCalType.DIRECT_CX_REVERSE, cr_tones, comp_tones)
    raise QiskitError(f'{repr(cr_sched)} is undefined pulse sequence. Check if this is a calibration for cross resonance operation.')