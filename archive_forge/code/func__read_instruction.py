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
def _read_instruction(file_obj, circuit, registers, custom_operations, version, vectors, use_symengine):
    if version < 5:
        instruction = formats.CIRCUIT_INSTRUCTION._make(struct.unpack(formats.CIRCUIT_INSTRUCTION_PACK, file_obj.read(formats.CIRCUIT_INSTRUCTION_SIZE)))
    else:
        instruction = formats.CIRCUIT_INSTRUCTION_V2._make(struct.unpack(formats.CIRCUIT_INSTRUCTION_V2_PACK, file_obj.read(formats.CIRCUIT_INSTRUCTION_V2_SIZE)))
    gate_name = file_obj.read(instruction.name_size).decode(common.ENCODE)
    label = file_obj.read(instruction.label_size).decode(common.ENCODE)
    condition_register = file_obj.read(instruction.condition_register_size).decode(common.ENCODE)
    qargs = []
    cargs = []
    params = []
    condition = None
    if version < 5 and instruction.has_condition or (version >= 5 and instruction.conditional_key == type_keys.Condition.TWO_TUPLE):
        condition = (_loads_register_param(condition_register, circuit, registers), instruction.condition_value)
    elif version >= 5 and instruction.conditional_key == type_keys.Condition.EXPRESSION:
        condition = value.read_value(file_obj, version, vectors, clbits=circuit.clbits, cregs=registers['c'], use_symengine=use_symengine)
    if circuit is not None:
        for _qarg in range(instruction.num_qargs):
            qarg = formats.CIRCUIT_INSTRUCTION_ARG._make(struct.unpack(formats.CIRCUIT_INSTRUCTION_ARG_PACK, file_obj.read(formats.CIRCUIT_INSTRUCTION_ARG_SIZE)))
            if qarg.type.decode(common.ENCODE) == 'c':
                raise TypeError('Invalid input carg prior to all qargs')
            qargs.append(circuit.qubits[qarg.size])
        for _carg in range(instruction.num_cargs):
            carg = formats.CIRCUIT_INSTRUCTION_ARG._make(struct.unpack(formats.CIRCUIT_INSTRUCTION_ARG_PACK, file_obj.read(formats.CIRCUIT_INSTRUCTION_ARG_SIZE)))
            if carg.type.decode(common.ENCODE) == 'q':
                raise TypeError('Invalid input qarg after all qargs')
            cargs.append(circuit.clbits[carg.size])
    for _param in range(instruction.num_parameters):
        type_key, data_bytes = common.read_generic_typed_data(file_obj)
        param = _loads_instruction_parameter(type_key, data_bytes, version, vectors, registers, circuit, use_symengine)
        params.append(param)
    if gate_name in {'Gate', 'Instruction', 'ControlledGate'}:
        inst_obj = _parse_custom_operation(custom_operations, gate_name, params, version, vectors, registers, use_symengine)
        inst_obj.condition = condition
        if instruction.label_size > 0:
            inst_obj.label = label
        if circuit is None:
            return inst_obj
        circuit._append(inst_obj, qargs, cargs)
        return None
    elif gate_name in custom_operations:
        inst_obj = _parse_custom_operation(custom_operations, gate_name, params, version, vectors, registers, use_symengine)
        inst_obj.condition = condition
        if instruction.label_size > 0:
            inst_obj.label = label
        if circuit is None:
            return inst_obj
        circuit._append(inst_obj, qargs, cargs)
        return None
    elif hasattr(library, gate_name):
        gate_class = getattr(library, gate_name)
    elif hasattr(circuit_mod, gate_name):
        gate_class = getattr(circuit_mod, gate_name)
    elif hasattr(controlflow, gate_name):
        gate_class = getattr(controlflow, gate_name)
    elif gate_name == 'Clifford':
        gate_class = Clifford
    else:
        raise AttributeError('Invalid instruction type: %s' % gate_name)
    if instruction.label_size <= 0:
        label = None
    if gate_name in {'IfElseOp', 'WhileLoopOp'}:
        gate = gate_class(condition, *params, label=label)
    elif version >= 5 and issubclass(gate_class, ControlledGate):
        if gate_name in {'MCPhaseGate', 'MCU1Gate', 'MCXGrayCode', 'MCXGate', 'MCXRecursive', 'MCXVChain'}:
            gate = gate_class(*params, instruction.num_ctrl_qubits, label=label)
        else:
            gate = gate_class(*params, label=label)
            if gate.num_ctrl_qubits != instruction.num_ctrl_qubits or gate.ctrl_state != instruction.ctrl_state:
                gate = gate.to_mutable()
                gate.num_ctrl_qubits = instruction.num_ctrl_qubits
                gate.ctrl_state = instruction.ctrl_state
        if condition:
            gate = gate.c_if(*condition)
    else:
        if gate_name in {'Initialize', 'StatePreparation'}:
            if isinstance(params[0], str):
                gate = gate_class(''.join((label for label in params)))
            elif instruction.num_parameters == 1:
                gate = gate_class(int(params[0].real), instruction.num_qargs)
            else:
                gate = gate_class(params)
        elif gate_name in {'UCRXGate', 'UCRYGate', 'UCRZGate', 'DiagonalGate'}:
            gate = gate_class(params)
        else:
            if gate_name == 'Barrier':
                params = [len(qargs)]
            elif gate_name in {'BreakLoopOp', 'ContinueLoopOp'}:
                params = [len(qargs), len(cargs)]
            if label is not None:
                if issubclass(gate_class, (SingletonInstruction, SingletonGate)):
                    gate = gate_class(*params, label=label)
                else:
                    gate = gate_class(*params)
                    gate.label = label
            else:
                gate = gate_class(*params)
        if condition:
            if not isinstance(gate, ControlFlowOp):
                gate = gate.c_if(*condition)
            else:
                gate.condition = condition
    if circuit is None:
        return gate
    if not isinstance(gate, Instruction):
        circuit.append(gate, qargs, cargs)
    else:
        circuit._append(CircuitInstruction(gate, qargs, cargs))
    return None