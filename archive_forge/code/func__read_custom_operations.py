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
def _read_custom_operations(file_obj, version, vectors):
    custom_operations = {}
    custom_definition_header = formats.CUSTOM_CIRCUIT_DEF_HEADER._make(struct.unpack(formats.CUSTOM_CIRCUIT_DEF_HEADER_PACK, file_obj.read(formats.CUSTOM_CIRCUIT_DEF_HEADER_SIZE)))
    if custom_definition_header.size > 0:
        for _ in range(custom_definition_header.size):
            if version < 5:
                data = formats.CUSTOM_CIRCUIT_INST_DEF._make(struct.unpack(formats.CUSTOM_CIRCUIT_INST_DEF_PACK, file_obj.read(formats.CUSTOM_CIRCUIT_INST_DEF_SIZE)))
            else:
                data = formats.CUSTOM_CIRCUIT_INST_DEF_V2._make(struct.unpack(formats.CUSTOM_CIRCUIT_INST_DEF_V2_PACK, file_obj.read(formats.CUSTOM_CIRCUIT_INST_DEF_V2_SIZE)))
            name = file_obj.read(data.gate_name_size).decode(common.ENCODE)
            type_str = data.type
            definition_circuit = None
            if data.custom_definition:
                def_binary = file_obj.read(data.size)
                if version < 3 or not name.startswith('###PauliEvolutionGate_'):
                    definition_circuit = common.data_from_binary(def_binary, read_circuit, version=version)
                elif name.startswith('###PauliEvolutionGate_'):
                    definition_circuit = common.data_from_binary(def_binary, _read_pauli_evolution_gate, version=version, vectors=vectors)
            if version < 5:
                data_payload = (type_str, data.num_qubits, data.num_clbits, definition_circuit)
            else:
                base_gate = file_obj.read(data.base_gate_size)
                data_payload = (type_str, data.num_qubits, data.num_clbits, definition_circuit, data.num_ctrl_qubits, data.ctrl_state, base_gate)
            custom_operations[name] = data_payload
    return custom_operations