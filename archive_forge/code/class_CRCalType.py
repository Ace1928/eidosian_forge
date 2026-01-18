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
class CRCalType(enum.Enum):
    """Estimated calibration type of backend cross resonance operations."""
    ECR_FORWARD = 'Echoed Cross Resonance corresponding to native operation'
    ECR_REVERSE = 'Echoed Cross Resonance reverse of native operation'
    ECR_CX_FORWARD = 'Echoed Cross Resonance CX corresponding to native operation'
    ECR_CX_REVERSE = 'Echoed Cross Resonance CX reverse of native operation'
    DIRECT_CX_FORWARD = 'Direct CX corresponding to native operation'
    DIRECT_CX_REVERSE = 'Direct CX reverse of native operation'