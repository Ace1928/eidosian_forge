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
def _read_common_layout(file_obj, header, circuit):
    registers = {name: QuantumRegister(len(v[1]), name) for name, v in _read_registers_v4(file_obj, header.extra_registers)['q'].items()}
    initial_layout = None
    initial_layout_virtual_bits = []
    for _ in range(header.initial_layout_size):
        virtual_bit = formats.INITIAL_LAYOUT_BIT._make(struct.unpack(formats.INITIAL_LAYOUT_BIT_PACK, file_obj.read(formats.INITIAL_LAYOUT_BIT_SIZE)))
        if virtual_bit.index == -1 and virtual_bit.register_size == -1:
            qubit = Qubit()
        else:
            register_name = file_obj.read(virtual_bit.register_size).decode(common.ENCODE)
            if register_name in registers:
                qubit = registers[register_name][virtual_bit.index]
            else:
                register = next(filter(lambda x, name=register_name: x.name == name, circuit.qregs))
                qubit = register[virtual_bit.index]
        initial_layout_virtual_bits.append(qubit)
    if initial_layout_virtual_bits:
        initial_layout = Layout.from_qubit_list(initial_layout_virtual_bits)
    input_qubit_mapping = None
    input_qubit_mapping_array = []
    for _ in range(header.input_mapping_size):
        input_qubit_mapping_array.append(struct.unpack('!I', file_obj.read(struct.calcsize('!I')))[0])
    if input_qubit_mapping_array:
        input_qubit_mapping = {}
        physical_bits = initial_layout.get_physical_bits()
        for index, bit in enumerate(input_qubit_mapping_array):
            input_qubit_mapping[physical_bits[bit]] = index
    final_layout = None
    final_layout_array = []
    for _ in range(header.final_layout_size):
        final_layout_array.append(struct.unpack('!I', file_obj.read(struct.calcsize('!I')))[0])
    if final_layout_array:
        layout_dict = {circuit.qubits[bit]: index for index, bit in enumerate(final_layout_array)}
        final_layout = Layout(layout_dict)
    circuit._layout = TranspileLayout(initial_layout, input_qubit_mapping, final_layout)