from typing import List, Tuple, Union
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
def _run_on_inverse_pairs(self, dag: DAGCircuit):
    """
        Run inverse gate pairs on `dag`.

        Args:
            dag: the directed acyclic graph to run on.
            inverse_gate_pairs: list of gates with inverse angles that cancel each other.

        Returns:
            DAGCircuit: Transformed DAG.
        """
    op_counts = dag.count_ops()
    if not self.inverse_gate_pairs_names.intersection(op_counts):
        return dag
    for pair in self.inverse_gate_pairs:
        gate_0_name = pair[0].name
        gate_1_name = pair[1].name
        if gate_0_name not in op_counts or gate_1_name not in op_counts:
            continue
        gate_cancel_runs = dag.collect_runs([gate_0_name, gate_1_name])
        for dag_nodes in gate_cancel_runs:
            i = 0
            while i < len(dag_nodes) - 1:
                if dag_nodes[i].qargs == dag_nodes[i + 1].qargs and dag_nodes[i].op == pair[0] and (dag_nodes[i + 1].op == pair[1]):
                    dag.remove_op_node(dag_nodes[i])
                    dag.remove_op_node(dag_nodes[i + 1])
                    i = i + 2
                elif dag_nodes[i].qargs == dag_nodes[i + 1].qargs and dag_nodes[i].op == pair[1] and (dag_nodes[i + 1].op == pair[0]):
                    dag.remove_op_node(dag_nodes[i])
                    dag.remove_op_node(dag_nodes[i + 1])
                    i = i + 2
                else:
                    i = i + 1
    return dag