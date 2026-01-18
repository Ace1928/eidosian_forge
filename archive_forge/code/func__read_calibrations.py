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
def _read_calibrations(file_obj, version, vectors, metadata_deserializer):
    calibrations = {}
    header = formats.CALIBRATION._make(struct.unpack(formats.CALIBRATION_PACK, file_obj.read(formats.CALIBRATION_SIZE)))
    for _ in range(header.num_cals):
        defheader = formats.CALIBRATION_DEF._make(struct.unpack(formats.CALIBRATION_DEF_PACK, file_obj.read(formats.CALIBRATION_DEF_SIZE)))
        name = file_obj.read(defheader.name_size).decode(common.ENCODE)
        qubits = tuple((struct.unpack('!q', file_obj.read(struct.calcsize('!q')))[0] for _ in range(defheader.num_qubits)))
        params = tuple((value.read_value(file_obj, version, vectors) for _ in range(defheader.num_params)))
        schedule = schedules.read_schedule_block(file_obj, version, metadata_deserializer)
        if name not in calibrations:
            calibrations[name] = {(qubits, params): schedule}
        else:
            calibrations[name][qubits, params] = schedule
    return calibrations