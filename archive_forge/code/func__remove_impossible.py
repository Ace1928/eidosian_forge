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