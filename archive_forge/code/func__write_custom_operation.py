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
def _write_custom_operation(file_obj, name, operation, custom_operations, use_symengine, version):
    type_key = type_keys.CircuitInstruction.assign(operation)
    has_definition = False
    size = 0
    data = None
    num_qubits = operation.num_qubits
    num_clbits = operation.num_clbits
    ctrl_state = 0
    num_ctrl_qubits = 0
    base_gate = None
    new_custom_instruction = []
    if type_key == type_keys.CircuitInstruction.PAULI_EVOL_GATE:
        has_definition = True
        data = common.data_to_binary(operation, _write_pauli_evolution_gate)
        size = len(data)
    elif type_key == type_keys.CircuitInstruction.CONTROLLED_GATE:
        has_definition = True
        operation.definition
        data = common.data_to_binary(operation._definition, write_circuit)
        size = len(data)
        num_ctrl_qubits = operation.num_ctrl_qubits
        ctrl_state = operation.ctrl_state
        base_gate = operation.base_gate
    elif type_key == type_keys.CircuitInstruction.ANNOTATED_OPERATION:
        has_definition = False
        base_gate = operation.base_op
    elif operation.definition is not None:
        has_definition = True
        data = common.data_to_binary(operation.definition, write_circuit)
        size = len(data)
    if base_gate is None:
        base_gate_raw = b''
    else:
        with io.BytesIO() as base_gate_buffer:
            new_custom_instruction = _write_instruction(base_gate_buffer, CircuitInstruction(base_gate, (), ()), custom_operations, {}, use_symengine, version)
            base_gate_raw = base_gate_buffer.getvalue()
    name_raw = name.encode(common.ENCODE)
    custom_operation_raw = struct.pack(formats.CUSTOM_CIRCUIT_INST_DEF_V2_PACK, len(name_raw), type_key, num_qubits, num_clbits, has_definition, size, num_ctrl_qubits, ctrl_state, len(base_gate_raw))
    file_obj.write(custom_operation_raw)
    file_obj.write(name_raw)
    if data:
        file_obj.write(data)
    file_obj.write(base_gate_raw)
    return new_custom_instruction