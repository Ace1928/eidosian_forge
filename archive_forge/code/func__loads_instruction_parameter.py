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
def _loads_instruction_parameter(type_key, data_bytes, version, vectors, registers, circuit, use_symengine):
    if type_key == type_keys.Program.CIRCUIT:
        param = common.data_from_binary(data_bytes, read_circuit, version=version)
    elif type_key == type_keys.Value.MODIFIER:
        param = common.data_from_binary(data_bytes, _read_modifier)
    elif type_key == type_keys.Container.RANGE:
        data = formats.RANGE._make(struct.unpack(formats.RANGE_PACK, data_bytes))
        param = range(data.start, data.stop, data.step)
    elif type_key == type_keys.Container.TUPLE:
        param = tuple(common.sequence_from_binary(data_bytes, _loads_instruction_parameter, version=version, vectors=vectors, registers=registers, circuit=circuit, use_symengine=use_symengine))
    elif type_key == type_keys.Value.INTEGER:
        param = struct.unpack('<q', data_bytes)[0]
    elif type_key == type_keys.Value.FLOAT:
        param = struct.unpack('<d', data_bytes)[0]
    elif type_key == type_keys.Value.REGISTER:
        param = _loads_register_param(data_bytes.decode(common.ENCODE), circuit, registers)
    else:
        clbits = circuit.clbits if circuit is not None else ()
        param = value.loads_value(type_key, data_bytes, version, vectors, clbits=clbits, cregs=registers['c'], use_symengine=use_symengine)
    return param