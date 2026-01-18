import collections
import copy
import itertools
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagdependency import DAGDependency
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag
from qiskit.utils import optionals as _optionals
def _incr_num_parameters(self, template):
    """
        Checks if template substitution would increase the number of
        parameters in the circuit.
        """
    template_params = set()
    for param_list in (node.op.params for node in template.get_nodes()):
        for param_exp in param_list:
            if isinstance(param_exp, ParameterExpression):
                template_params.update(param_exp.parameters)
    circuit_params = set()
    for param_list in (node.op.params for node in self.circuit_dag_dep.get_nodes()):
        for param_exp in param_list:
            if isinstance(param_exp, ParameterExpression):
                circuit_params.update(param_exp.parameters)
    return len(template_params) > len(circuit_params)