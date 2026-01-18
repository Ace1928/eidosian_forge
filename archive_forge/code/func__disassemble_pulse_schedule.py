from typing import Any, Dict, List, NewType, Tuple, Union
import collections
import math
from qiskit import pulse
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.qobj import PulseQobjInstruction
from qiskit.qobj.converters import QobjToInstructionConverter
def _disassemble_pulse_schedule(qobj) -> PulseModule:
    run_config = qobj.config.to_dict()
    run_config.pop('pulse_library')
    qubit_lo_freq = run_config.get('qubit_lo_freq')
    if qubit_lo_freq:
        run_config['qubit_lo_freq'] = [freq * 1000000000.0 for freq in qubit_lo_freq]
    meas_lo_freq = run_config.get('meas_lo_freq')
    if meas_lo_freq:
        run_config['meas_lo_freq'] = [freq * 1000000000.0 for freq in meas_lo_freq]
    user_qobj_header = qobj.header.to_dict()
    schedule_los = []
    for program in qobj.experiments:
        program_los = {}
        if hasattr(program, 'config'):
            if hasattr(program.config, 'qubit_lo_freq'):
                for i, lo in enumerate(program.config.qubit_lo_freq):
                    program_los[pulse.DriveChannel(i)] = lo * 1000000000.0
            if hasattr(program.config, 'meas_lo_freq'):
                for i, lo in enumerate(program.config.meas_lo_freq):
                    program_los[pulse.MeasureChannel(i)] = lo * 1000000000.0
        schedule_los.append(program_los)
    if any(schedule_los):
        run_config['schedule_los'] = schedule_los
    return PulseModule((_experiments_to_schedules(qobj), run_config, user_qobj_header))