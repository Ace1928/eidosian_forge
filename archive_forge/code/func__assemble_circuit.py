import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from qiskit.assembler.run_config import RunConfig
from qiskit.assembler.assemble_schedules import _assemble_instructions as _assemble_schedule
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classicalregister import Clbit
from qiskit.exceptions import QiskitError
from qiskit.qobj import (
from qiskit.utils.parallel import parallel_map
def _assemble_circuit(circuit: QuantumCircuit, run_config: RunConfig) -> Tuple[QasmQobjExperiment, Optional[PulseLibrary]]:
    """Assemble one circuit.

    Args:
        circuit: circuit to assemble
        run_config: configuration of the runtime environment

    Returns:
        One experiment for the QasmQobj, and pulse library for pulse gates (which could be None)

    Raises:
        QiskitError: when the circuit has unit other than 'dt'.
    """
    if circuit.unit != 'dt':
        raise QiskitError(f"Unable to assemble circuit with unit '{circuit.unit}', which must be 'dt'.")
    num_qubits = 0
    memory_slots = 0
    qubit_labels = []
    clbit_labels = []
    qreg_sizes = []
    creg_sizes = []
    for qreg in circuit.qregs:
        qreg_sizes.append([qreg.name, qreg.size])
        for j in range(qreg.size):
            qubit_labels.append([qreg.name, j])
        num_qubits += qreg.size
    for creg in circuit.cregs:
        creg_sizes.append([creg.name, creg.size])
        for j in range(creg.size):
            clbit_labels.append([creg.name, j])
        memory_slots += creg.size
    qubit_indices = {qubit: idx for idx, qubit in enumerate(circuit.qubits)}
    clbit_indices = {clbit: idx for idx, clbit in enumerate(circuit.clbits)}
    metadata = circuit.metadata
    if metadata is None:
        metadata = {}
    header = QobjExperimentHeader(qubit_labels=qubit_labels, n_qubits=num_qubits, qreg_sizes=qreg_sizes, clbit_labels=clbit_labels, memory_slots=memory_slots, creg_sizes=creg_sizes, name=circuit.name, global_phase=float(circuit.global_phase), metadata=metadata)
    config = QasmQobjExperimentConfig(n_qubits=num_qubits, memory_slots=memory_slots)
    calibrations, pulse_library = _assemble_pulse_gates(circuit, run_config)
    if calibrations:
        config.calibrations = calibrations
    is_conditional_experiment = any((getattr(instruction.operation, 'condition', None) for instruction in circuit.data))
    max_conditional_idx = 0
    instructions = []
    for op_context in circuit.data:
        instruction = op_context.operation.assemble()
        qargs = op_context.qubits
        cargs = op_context.clbits
        if qargs:
            instruction.qubits = [qubit_indices[qubit] for qubit in qargs]
        if cargs:
            instruction.memory = [clbit_indices[clbit] for clbit in cargs]
            if instruction.name == 'measure' and is_conditional_experiment:
                instruction.register = [clbit_indices[clbit] for clbit in cargs]
        if hasattr(instruction, '_condition'):
            ctrl_reg, ctrl_val = instruction._condition
            mask = 0
            val = 0
            if isinstance(ctrl_reg, Clbit):
                mask = 1 << clbit_indices[ctrl_reg]
                val = (ctrl_val & 1) << clbit_indices[ctrl_reg]
            else:
                for clbit in clbit_indices:
                    if clbit in ctrl_reg:
                        mask |= 1 << clbit_indices[clbit]
                        val |= (ctrl_val >> list(ctrl_reg).index(clbit) & 1) << clbit_indices[clbit]
            conditional_reg_idx = memory_slots + max_conditional_idx
            conversion_bfunc = QasmQobjInstruction(name='bfunc', mask='0x%X' % mask, relation='==', val='0x%X' % val, register=conditional_reg_idx)
            instructions.append(conversion_bfunc)
            instruction.conditional = conditional_reg_idx
            max_conditional_idx += 1
            del instruction._condition
        instructions.append(instruction)
    return (QasmQobjExperiment(instructions=instructions, header=header, config=config), pulse_library)