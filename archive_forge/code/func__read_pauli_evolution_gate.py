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
def _read_pauli_evolution_gate(file_obj, version, vectors):
    pauli_evolution_def = formats.PAULI_EVOLUTION_DEF._make(struct.unpack(formats.PAULI_EVOLUTION_DEF_PACK, file_obj.read(formats.PAULI_EVOLUTION_DEF_SIZE)))
    if pauli_evolution_def.operator_size != 1 and pauli_evolution_def.standalone_op:
        raise ValueError("Can't have a standalone operator with {pauli_evolution_raw[0]} operators in the payload")
    operator_list = []
    for _ in range(pauli_evolution_def.operator_size):
        op_elem = formats.SPARSE_PAULI_OP_LIST_ELEM._make(struct.unpack(formats.SPARSE_PAULI_OP_LIST_ELEM_PACK, file_obj.read(formats.SPARSE_PAULI_OP_LIST_ELEM_SIZE)))
        op_raw_data = common.data_from_binary(file_obj.read(op_elem.size), np.load)
        operator_list.append(SparsePauliOp.from_list(op_raw_data))
    if pauli_evolution_def.standalone_op:
        pauli_op = operator_list[0]
    else:
        pauli_op = operator_list
    time = value.loads_value(pauli_evolution_def.time_type, file_obj.read(pauli_evolution_def.time_size), version=version, vectors=vectors)
    synth_data = json.loads(file_obj.read(pauli_evolution_def.synth_method_size))
    synthesis = getattr(evo_synth, synth_data['class'])(**synth_data['settings'])
    return_gate = library.PauliEvolutionGate(pauli_op, time=time, synthesis=synthesis)
    return return_gate