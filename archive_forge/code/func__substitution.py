import collections
import copy
import itertools
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagdependency import DAGDependency
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag
from qiskit.utils import optionals as _optionals
def _substitution(self):
    """
        From the list of maximal matches, it chooses which one will be used and gives the necessary
        details for each substitution(template inverse, predecessors of the match).
        """
    while self.match_stack:
        current = self.match_stack.pop(0)
        current_match = current.match
        current_qubit = current.qubit
        current_clbit = current.clbit
        template_sublist = [x[0] for x in current_match]
        circuit_sublist = [x[1] for x in current_match]
        circuit_sublist.sort()
        template = self._attempt_bind(template_sublist, circuit_sublist)
        if template is None or self._incr_num_parameters(template):
            continue
        template_list = range(0, self.template_dag_dep.size())
        template_complement = list(set(template_list) - set(template_sublist))
        if self._rules(circuit_sublist, template_sublist, template_complement):
            template_sublist_inverse = self._template_inverse(template_list, template_sublist, template_complement)
            config = SubstitutionConfig(circuit_sublist, template_sublist_inverse, [], current_qubit, template, current_clbit)
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
    self.unmatched_list = sorted(set(range(0, self.circuit_dag_dep.size())) - set(circuit_list))