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
def _find_adjoining_run(dag, runs, run, front=True):
    """
        Finds the run which abuts `run` from the front (or the rear if `front == False`), separated
        by a blocking node.

        Returns a pair of the abutting multiqubit gate and the run which it separates from this
        one. The next run can be the empty list `[]` if it is absent.
        """
    edge_node = run[0] if front else run[-1]
    blocker = next(dag.predecessors(edge_node) if front else dag.successors(edge_node))
    possibilities = dag.predecessors(blocker) if front else dag.successors(blocker)
    adjoining_run = []
    for possibility in possibilities:
        if isinstance(possibility, DAGOpNode) and possibility.qargs == edge_node.qargs:
            adjoining_run = []
            for single_run in runs:
                if len(single_run) != 0 and single_run[0].qargs == possibility.qargs:
                    if possibility in single_run:
                        adjoining_run = single_run
                        break
            break
    return (blocker, adjoining_run)