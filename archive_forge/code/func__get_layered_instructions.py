import re
from collections import OrderedDict
import numpy as np
from qiskit.circuit import (
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import ClassicalRegister, QuantumCircuit, Qubit, ControlFlowOp
from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier, PowerModifier
from qiskit.circuit.tools import pi_check
from qiskit.converters import circuit_to_dag
from qiskit.utils import optionals as _optionals
from ..exceptions import VisualizationError
def _get_layered_instructions(circuit, reverse_bits=False, justify=None, idle_wires=True, wire_order=None, wire_map=None):
    """
    Given a circuit, return a tuple (qubits, clbits, nodes) where
    qubits and clbits are the quantum and classical registers
    in order (based on reverse_bits or wire_order) and nodes
    is a list of DAGOpNodes.

    Args:
        circuit (QuantumCircuit): From where the information is extracted.
        reverse_bits (bool): If true the order of the bits in the registers is
            reversed.
        justify (str) : `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        idle_wires (bool): Include idle wires. Default is True.
        wire_order (list): A list of ints that modifies the order of the bits

    Returns:
        Tuple(list,list,list): To be consumed by the visualizer directly.

    Raises:
        VisualizationError: if both reverse_bits and wire_order are entered.
    """
    if justify:
        justify = justify.lower()
    justify = justify if justify in ('right', 'none') else 'left'
    if wire_map is not None:
        qubits = [bit for bit in wire_map if isinstance(bit, Qubit)]
    else:
        qubits = circuit.qubits.copy()
    clbits = circuit.clbits.copy()
    nodes = []
    measure_map = OrderedDict([(c, -1) for c in clbits])
    if reverse_bits and wire_order is not None:
        raise VisualizationError('Cannot set both reverse_bits and wire_order in the same drawing.')
    if reverse_bits:
        qubits.reverse()
        clbits.reverse()
    elif wire_order is not None:
        new_qubits = []
        new_clbits = []
        for bit in wire_order:
            if bit < len(qubits):
                new_qubits.append(qubits[bit])
            else:
                new_clbits.append(clbits[bit - len(qubits)])
        qubits = new_qubits
        clbits = new_clbits
    dag = circuit_to_dag(circuit)
    dag.qubits = qubits
    dag.clbits = clbits
    if justify == 'none':
        for node in dag.topological_op_nodes():
            nodes.append([node])
    else:
        nodes = _LayerSpooler(dag, justify, measure_map)
    if not idle_wires:
        for wire in dag.idle_wires(ignore=['barrier', 'delay']):
            if wire in qubits:
                qubits.remove(wire)
            if wire in clbits:
                clbits.remove(wire)
    nodes = [[node for node in layer if any((q in qubits for q in node.qargs))] for layer in nodes]
    return (qubits, clbits, nodes)