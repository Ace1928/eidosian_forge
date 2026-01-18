import time
import logging
from functools import singledispatchmethod
from itertools import zip_longest
from collections import defaultdict
import rustworkx
from qiskit.circuit import (
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.equivalence import Key, NodeData
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
def _get_example_gates(source_dag):

    def recurse(dag, example_gates=None):
        example_gates = example_gates or {}
        for node in dag.op_nodes():
            example_gates[node.op.name, node.op.num_qubits] = node.op
            if isinstance(node.op, ControlFlowOp):
                for block in node.op.blocks:
                    example_gates = recurse(circuit_to_dag(block), example_gates)
        return example_gates
    return recurse(source_dag)