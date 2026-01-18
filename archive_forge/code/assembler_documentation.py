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
Verifies that there is a single common set of parameters shared between
    all circuits and all parameter binds in the run_config. Returns an expanded
    list of circuits (if parameterized) with all parameters bound, and a copy of
    the run_config with parameter_binds cleared.

    If neither the circuits nor the run_config specify parameters, the two are
    returned unmodified.

    Raises:
        QiskitError: if run_config parameters are not compatible with circuit parameters

    Returns:
        Tuple(List[QuantumCircuit], RunConfig):
          - List of input circuits expanded and with parameters bound
          - RunConfig with parameter_binds removed
    