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
def _qobj_to_circuit_cals(qobj, pulse_lib):
    """Return circuit calibrations dictionary from qobj/exp config calibrations."""
    qobj_cals = qobj.config.calibrations.to_dict()['gates']
    converter = QobjToInstructionConverter(pulse_lib)
    qc_cals = {}
    for gate in qobj_cals:
        config = (tuple(gate['qubits']), tuple(gate['params']))
        cal = {config: pulse.Schedule(name='{} {} {}'.format(gate['name'], str(gate['params']), str(gate['qubits'])))}
        for instruction in gate['instructions']:
            qobj_instruction = PulseQobjInstruction.from_dict(instruction)
            schedule = converter(qobj_instruction)
            cal[config] = cal[config].insert(schedule.ch_start_time(), schedule)
        if gate['name'] in qc_cals:
            qc_cals[gate['name']].update(cal)
        else:
            qc_cals[gate['name']] = cal
    return qc_cals