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
class TemplateSubstitution:
    """Class to run the substitution algorithm from the list of maximal matches."""

    def __init__(self, max_matches, circuit_dag, template_dag, custom_quantum_cost=None):
        """
        Initialize TemplateSubstitution with necessary arguments.
        Args:
            max_matches (list(int)): list of maximal matches obtained from the running the pattern matching algorithm.
            circuit_dag (.CommutationDAG): circuit in the dag dependency form.
            template_dag (.CommutationDAG): template in the dag dependency form.
            custom_quantum_cost (dict): Optional, quantum cost that overrides the default cost dictionnary.
        """
        self.match_stack = max_matches
        self.circuit_dag = circuit_dag
        self.template_dag = template_dag
        self.substitution_list = []
        self.unmatched_list = []
        if custom_quantum_cost is not None:
            self.quantum_cost = dict(custom_quantum_cost)
        else:
            self.quantum_cost = {'Identity': 0, 'PauliX': 1, 'PauliY': 1, 'PauliZ': 1, 'RX': 1, 'RY': 1, 'RZ': 1, 'Hadamard': 1, 'T': 1, 'Adjoint(T)': 1, 'S': 1, 'Adjoint(S)': 1, 'CNOT': 2, 'CZ': 4, 'SWAP': 6, 'CSWAP': 63, 'Toffoli': 21}

    def _pred_block(self, circuit_sublist, index):
        """It returns the predecessors of a given part of the circuit.
        Args:
            circuit_sublist (list): list of the gates matched in the circuit.
            index (int): Index of the group of matches.
        Returns:
            list: List of predecessors of the current match circuit configuration.
        """
        predecessors = set()
        for node_id in circuit_sublist:
            predecessors = predecessors | set(self.circuit_dag.get_node(node_id).predecessors)
        exclude = set()
        for elem in self.substitution_list[:index]:
            exclude = exclude | set(elem.circuit_config) | set(elem.pred_block)
        pred = list(predecessors - set(circuit_sublist) - exclude)
        pred.sort()
        return pred

    def _quantum_cost(self, left, right):
        """Compare the two parts of the template and returns True if the quantum cost is reduced.
        Args:
            left (list): list of matched nodes in the template.
            right (list): list of nodes to be replaced.
        Returns:
            bool: True if the quantum cost is reduced
        """
        cost_left = 0
        for i in left:
            cost_left += self.quantum_cost[self.template_dag.get_node(i).op.name]
        cost_right = 0
        for j in right:
            cost_right += self.quantum_cost[self.template_dag.get_node(j).op.name]
        return cost_left > cost_right

    def _rules(self, circuit_sublist, template_sublist, template_complement):
        """Set of rules to decide whether the match is to be substitute or not.
        Args:
            circuit_sublist (list): list of the gates matched in the circuit.
            template_sublist (list): list of matched nodes in the template.
            template_complement (list): list of gates not matched in the template.
        Returns:
            bool: True if the match respects the given rule for replacement, False otherwise.
        """
        if self._quantum_cost(template_sublist, template_complement):
            for elem in circuit_sublist:
                for config in self.substitution_list:
                    if any((elem == x for x in config.circuit_config)):
                        return False
            return True
        return False

    def _template_inverse(self, template_list, template_sublist, template_complement):
        """The template circuit realizes the identity operator, then given the list of matches in the template,
        it returns the inverse part of the template that will be replaced.
        Args:
            template_list (list): list of all gates in the template.
            template_sublist (list): list of the gates matched in the circuit.
            template_complement  (list): list of gates not matched in the template.
        Returns:
            list(int): the template inverse part that will substitute the circuit match.
        """
        inverse = template_complement
        left = []
        right = []
        pred = set()
        for index in template_sublist:
            pred = pred | set(self.template_dag.get_node(index).predecessors)
        pred = list(pred - set(template_sublist))
        succ = set()
        for index in template_sublist:
            succ = succ | set(self.template_dag.get_node(index).successors)
        succ = list(succ - set(template_sublist))
        comm = list(set(template_list) - set(pred) - set(succ))
        for elem in inverse:
            if elem in pred:
                left.append(elem)
            elif elem in succ:
                right.append(elem)
            elif elem in comm:
                right.append(elem)
        left.sort()
        right.sort()
        left.reverse()
        right.reverse()
        total = left + right
        return total

    def _substitution_sort(self):
        """Sort the substitution list."""
        ordered = False
        while not ordered:
            ordered = self._permutation()

    def _permutation(self):
        """Permute two groups of matches if first one has predecessors in the second one.
        Returns:
            bool: True if the matches groups are in the right order, False otherwise.
        """
        for scenario in self.substitution_list:
            predecessors = set()
            for match in scenario.circuit_config:
                predecessors = predecessors | set(self.circuit_dag.get_node(match).predecessors)
            predecessors = predecessors - set(scenario.circuit_config)
            index = self.substitution_list.index(scenario)
            for scenario_b in self.substitution_list[index:]:
                if set(scenario_b.circuit_config) & predecessors:
                    index1 = self.substitution_list.index(scenario)
                    index2 = self.substitution_list.index(scenario_b)
                    scenario_pop = self.substitution_list.pop(index2)
                    self.substitution_list.insert(index1, scenario_pop)
                    return False
        return True

    def _remove_impossible(self):
        """Remove matched groups if they both have predecessors in the other one, they are not compatible."""
        list_predecessors = []
        remove_list = []
        for scenario in self.substitution_list:
            predecessors = set()
            for index in scenario.circuit_config:
                predecessors = predecessors | set(self.circuit_dag.get_node(index).predecessors)
            list_predecessors.append(predecessors)
        for scenario_a in self.substitution_list:
            if scenario_a in remove_list:
                continue
            index_a = self.substitution_list.index(scenario_a)
            circuit_a = scenario_a.circuit_config
            for scenario_b in self.substitution_list[index_a + 1:]:
                if scenario_b in remove_list:
                    continue
                index_b = self.substitution_list.index(scenario_b)
                circuit_b = scenario_b.circuit_config
                if set(circuit_a) & list_predecessors[index_b] and set(circuit_b) & list_predecessors[index_a]:
                    remove_list.append(scenario_b)
        if remove_list:
            self.substitution_list = [scenario for scenario in self.substitution_list if scenario not in remove_list]

    def substitution(self):
        """From the list of maximal matches, it chooses which one will be used and gives the necessary details for
        each substitution(template inverse, predecessors of the match).
        """
        while self.match_stack:
            current = self.match_stack.pop(0)
            current_match = current.match
            current_qubit = current.qubit
            template_sublist = [x[0] for x in current_match]
            circuit_sublist = [x[1] for x in current_match]
            circuit_sublist.sort()
            template_list = range(0, self.template_dag.size)
            template_complement = list(set(template_list) - set(template_sublist))
            if self._rules(circuit_sublist, template_sublist, template_complement):
                template_sublist_inverse = self._template_inverse(template_list, template_sublist, template_complement)
                config = SubstitutionConfig(circuit_sublist, template_sublist_inverse, [], current_qubit, self.template_dag)
                self.substitution_list.append(config)
        self._remove_impossible()
        self.substitution_list.sort(key=lambda x: x.circuit_config[0])
        self._substitution_sort()
        for scenario in self.substitution_list:
            index = self.substitution_list.index(scenario)
            scenario.pred_block = self._pred_block(scenario.circuit_config, index)
        circuit_list = []
        for elem in self.substitution_list:
            circuit_list = circuit_list + elem.circuit_config + elem.pred_block
        self.unmatched_list = sorted(list(set(range(0, self.circuit_dag.size)) - set(circuit_list)))