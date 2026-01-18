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
def _read_layout_v2(file_obj, circuit):
    header = formats.LAYOUT_V2._make(struct.unpack(formats.LAYOUT_V2_PACK, file_obj.read(formats.LAYOUT_V2_SIZE)))
    if not header.exists:
        return
    _read_common_layout(file_obj, header, circuit)
    if header.input_qubit_count >= 0:
        circuit._layout._input_qubit_count = header.input_qubit_count
        circuit._layout._output_qubit_list = circuit.qubits