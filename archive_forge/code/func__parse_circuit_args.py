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
def _parse_circuit_args(parameter_binds, backend, meas_level, meas_return, parametric_pulses, **run_config):
    """Build a circuit RunConfig replacing unset arguments with defaults derived from the `backend`.
    See `assemble` for more information on the required arguments.

    Returns:
        RunConfig: a run config, which is a standardized object that configures the qobj
            and determines the runtime environment.
    """
    parameter_binds = parameter_binds or []
    run_config_dict = dict(parameter_binds=parameter_binds, **run_config)
    if parametric_pulses is None:
        if backend:
            run_config_dict['parametric_pulses'] = getattr(backend.configuration(), 'parametric_pulses', [])
        else:
            run_config_dict['parametric_pulses'] = []
    else:
        run_config_dict['parametric_pulses'] = parametric_pulses
    if meas_level:
        run_config_dict['meas_level'] = meas_level
        if meas_level != MeasLevel.CLASSIFIED:
            run_config_dict['meas_return'] = meas_return
    run_config = RunConfig(**{k: v for k, v in run_config_dict.items() if v is not None})
    return run_config