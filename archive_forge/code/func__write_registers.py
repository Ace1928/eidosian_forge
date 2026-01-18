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
def _write_registers(file_obj, in_circ_regs, full_bits):
    bitmap = {bit: index for index, bit in enumerate(full_bits)}
    out_circ_regs = set()
    for bit in full_bits:
        if bit._register is not None and bit._register not in in_circ_regs:
            out_circ_regs.add(bit._register)
    for regs, is_in_circuit in [(in_circ_regs, True), (out_circ_regs, False)]:
        for reg in regs:
            standalone = all((bit._register is reg for bit in reg))
            reg_name = reg.name.encode(common.ENCODE)
            reg_type = reg.prefix.encode(common.ENCODE)
            file_obj.write(struct.pack(formats.REGISTER_V4_PACK, reg_type, standalone, reg.size, len(reg_name), is_in_circuit))
            file_obj.write(reg_name)
            REGISTER_ARRAY_PACK = '!%sq' % reg.size
            bit_indices = []
            for bit in reg:
                bit_indices.append(bitmap.get(bit, -1))
            file_obj.write(struct.pack(REGISTER_ARRAY_PACK, *bit_indices))
    return len(in_circ_regs) + len(out_circ_regs)