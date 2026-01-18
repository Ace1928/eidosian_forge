from copy import copy
import logging
from collections import deque
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import CXGate, RZXGate
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
@staticmethod
def _replace_subdag(dag, old_run, new_dag):
    """
        Replaces a nonempty sequence `old_run` of `DAGNode`s, assumed to be a complete chain in
        `dag`, with the circuit `new_circ`.
        """
    node_map = dag.substitute_node_with_dag(old_run[0], new_dag)
    for node in old_run[1:]:
        dag.remove_op_node(node)
    spliced_run = [node_map[node._node_id] for node in new_dag.topological_op_nodes()]
    mov_list(old_run, spliced_run)