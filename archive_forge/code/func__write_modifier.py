from collections import defaultdict
import io
import json
import struct
import uuid
import warnings
import numpy as np
from qiskit import circuit as circuit_mod
from qiskit.circuit import library, controlflow, CircuitInstruction, ControlFlowOp
from qiskit.circuit.classical import expr
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.singleton import SingletonInstruction, SingletonGate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.annotated_operation import (
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.qpy import common, formats, type_keys
from qiskit.qpy.binary_io import value, schedules
from qiskit.quantum_info.operators import SparsePauliOp, Clifford
from qiskit.synthesis import evolution as evo_synth
from qiskit.transpiler.layout import Layout, TranspileLayout
def _write_modifier(file_obj, modifier):
    if isinstance(modifier, InverseModifier):
        type_key = b'i'
        num_ctrl_qubits = 0
        ctrl_state = 0
        power = 0.0
    elif isinstance(modifier, ControlModifier):
        type_key = b'c'
        num_ctrl_qubits = modifier.num_ctrl_qubits
        ctrl_state = modifier.ctrl_state
        power = 0.0
    elif isinstance(modifier, PowerModifier):
        type_key = b'p'
        num_ctrl_qubits = 0
        ctrl_state = 0
        power = modifier.power
    else:
        raise TypeError('Unsupported modifier.')
    modifier_data = struct.pack(formats.MODIFIER_DEF_PACK, type_key, num_ctrl_qubits, ctrl_state, power)
    file_obj.write(modifier_data)