import copy
import itertools
from collections import OrderedDict
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
from pennylane import adjoint
from pennylane.ops.qubit.attributes import symmetric_over_all_wires
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires
class ForwardMatch:
    """
    Class to apply pattern matching in the forward direction.
    """

    def __init__(self, circuit_dag, pattern_dag, node_id_c, node_id_p, wires, control_wires, target_wires):
        """
        Create the ForwardMatch class.
        Args:
            circuit_dag (.CommutationDAG): circuit as commutation DAG.
            pattern_dag (.CommutationDAG): pattern as commutation DAG.
            node_id_c (int): index of the first gate matched in the circuit.
            node_id_p (int): index of the first gate matched in the pattern.
        """
        self.circuit_dag = circuit_dag
        self.pattern_dag = pattern_dag
        self.node_id_c = node_id_c
        self.node_id_p = node_id_p
        self.wires = wires
        self.control_wires = control_wires
        self.target_wires = target_wires
        self.successors_to_visit = [None] * circuit_dag.size
        self.circuit_blocked = [None] * circuit_dag.size
        self.circuit_matched_with = [None] * circuit_dag.size
        self.pattern_matched_with = [None] * pattern_dag.size
        self.updated_qubits = []
        self.match = []
        self.candidates = []
        self.matched_nodes_list = []

    def _init_successors_to_visit(self):
        """
        Initialize the list of successors to visit.
        """
        for i in range(0, self.circuit_dag.size):
            if i == self.node_id_c:
                self.successors_to_visit[i] = self.circuit_dag.direct_successors(i)
            else:
                self.successors_to_visit[i] = []

    def _init_matched_with_circuit(self):
        """
        Initialize the list of corresponding matches in the pattern for the circuit.
        """
        for i in range(0, self.circuit_dag.size):
            if i == self.node_id_c:
                self.circuit_matched_with[i] = [self.node_id_p]
            else:
                self.circuit_matched_with[i] = []

    def _init_matched_with_pattern(self):
        """
        Initialize the list of corresponding matches in the circuit for the pattern.
        """
        for i in range(0, self.pattern_dag.size):
            if i == self.node_id_p:
                self.pattern_matched_with[i] = [self.node_id_c]
            else:
                self.pattern_matched_with[i] = []

    def _init_is_blocked_circuit(self):
        """
        Initialize the list of blocked nodes in the circuit.
        """
        for i in range(0, self.circuit_dag.size):
            self.circuit_blocked[i] = False

    def _init_list_match(self):
        """
        Initialize the list of matched nodes between the circuit and the pattern with the first match found.
        """
        self.match.append([self.node_id_p, self.node_id_c])

    def _find_forward_candidates(self, node_id_p):
        """Find the candidate nodes to be matched in the pattern for a given node in the pattern.
        Args:
            node_id_p (int): Node ID in pattern.
        """
        matches = [i[0] for i in self.match]
        pred = matches.copy()
        if len(pred) > 1:
            pred.sort()
        pred.remove(node_id_p)
        if self.pattern_dag.direct_successors(node_id_p):
            maximal_index = self.pattern_dag.direct_successors(node_id_p)[-1]
            for elem in pred:
                if elem > maximal_index:
                    pred.remove(elem)
        block = []
        for node_id in pred:
            for dir_succ in self.pattern_dag.direct_successors(node_id):
                if dir_succ not in matches:
                    succ = self.pattern_dag.successors(dir_succ)
                    block = block + succ
        self.candidates = list(set(self.pattern_dag.direct_successors(node_id_p)) - set(matches) - set(block))

    def _init_matched_nodes(self):
        """
        Initialize the list of current matched nodes.
        """
        self.matched_nodes_list.append([self.node_id_c, self.circuit_dag.get_node(self.node_id_c), self.successors_to_visit[self.node_id_c]])

    def _get_node_forward(self, list_id):
        """
        Return node and successors from the matched_nodes_list for a given ID.
        Args:
            list_id (int): considered list id of the desired node.
        Returns:
            CommutationDAGNode: Node from the matched_node_list.
            list(int): List of successors.
        """
        node = self.matched_nodes_list[list_id][1]
        succ = self.matched_nodes_list[list_id][2]
        return (node, succ)

    def _remove_node_forward(self, list_id):
        """Remove a node of the current matched_nodes_list for a given ID.
        Args:
            list_id (int): considered list id of the desired node.
        """
        self.matched_nodes_list.pop(list_id)

    def run_forward_match(self):
        """Apply the forward match algorithm and returns the list of matches given an initial match
        and a qubits configuration.
        """
        self._init_successors_to_visit()
        self._init_matched_with_circuit()
        self._init_matched_with_pattern()
        self._init_is_blocked_circuit()
        self._init_list_match()
        self._init_matched_nodes()
        while self.matched_nodes_list:
            v_first, successors_to_visit = self._get_node_forward(0)
            self._remove_node_forward(0)
            if not successors_to_visit:
                continue
            label = successors_to_visit[0]
            v = [label, self.circuit_dag.get_node(label)]
            successors_to_visit.pop(0)
            self.matched_nodes_list.append([v_first.node_id, v_first, successors_to_visit])
            self.matched_nodes_list.sort(key=lambda x: x[2])
            if self.circuit_blocked[v[0]] | (self.circuit_matched_with[v[0]] != []):
                continue
            self._find_forward_candidates(self.circuit_matched_with[v_first.node_id][0])
            match = False
            for i in self.candidates:
                if match:
                    break
                node_circuit = self.circuit_dag.get_node(label)
                node_pattern = self.pattern_dag.get_node(i)
                if len(self.wires[label]) != len(node_pattern.wires) or set(self.wires[label]) != set(node_pattern.wires) or node_circuit.op.name != node_pattern.op.name:
                    continue
                if _compare_operation_without_qubits(node_circuit, node_pattern):
                    if _compare_qubits(node_circuit, self.wires[label], self.target_wires[label], self.control_wires[label], node_pattern.wires, node_pattern.control_wires, node_pattern.target_wires):
                        self.circuit_matched_with[label] = [i]
                        self.pattern_matched_with[i] = [label]
                        self.match.append([i, label])
                        potential = self.circuit_dag.direct_successors(label)
                        for potential_id in potential:
                            if self.circuit_blocked[potential_id] | (self.circuit_matched_with[potential_id] != []):
                                potential.remove(potential_id)
                        sorted_potential = sorted(potential)
                        successorstovisit = sorted_potential
                        self.matched_nodes_list.append([v[0], v[1], successorstovisit])
                        self.matched_nodes_list.sort(key=lambda x: x[2])
                        match = True
                        continue
            if not match:
                self.circuit_blocked[label] = True
                for succ in v[1].successors:
                    self.circuit_blocked[succ] = True