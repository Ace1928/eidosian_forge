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
def _disassemble_circuit(qobj) -> CircuitModule:
    run_config = qobj.config.to_dict()
    qubit_lo_freq = run_config.get('qubit_lo_freq', [])
    if qubit_lo_freq:
        run_config['qubit_lo_freq'] = [freq * 1000000000.0 for freq in qubit_lo_freq]
    meas_lo_freq = run_config.get('meas_lo_freq', [])
    if meas_lo_freq:
        run_config['meas_lo_freq'] = [freq * 1000000000.0 for freq in meas_lo_freq]
    user_qobj_header = qobj.header.to_dict()
    return CircuitModule((_experiments_to_circuits(qobj), run_config, user_qobj_header))