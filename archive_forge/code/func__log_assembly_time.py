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
def _log_assembly_time(start_time, end_time):
    log_msg = 'Total Assembly Time - %.5f (ms)' % ((end_time - start_time) * 1000)
    logger.info(log_msg)