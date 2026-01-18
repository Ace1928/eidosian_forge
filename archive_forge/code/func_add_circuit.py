import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
def add_circuit(self, arcs: Sequence[ArcT]) -> Constraint:
    """Adds Circuit(arcs).

        Adds a circuit constraint from a sparse list of arcs that encode the graph.

        A circuit is a unique Hamiltonian path in a subgraph of the total
        graph. In case a node 'i' is not in the path, then there must be a
        loop arc 'i -> i' associated with a true literal. Otherwise
        this constraint will fail.

        Args:
          arcs: a list of arcs. An arc is a tuple (source_node, destination_node,
            literal). The arc is selected in the circuit if the literal is true.
            Both source_node and destination_node must be integers between 0 and the
            number of nodes - 1.

        Returns:
          An instance of the `Constraint` class.

        Raises:
          ValueError: If the list of arcs is empty.
        """
    if not arcs:
        raise ValueError('add_circuit expects a non-empty array of arcs')
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    for arc in arcs:
        tail = cmh.assert_is_int32(arc[0])
        head = cmh.assert_is_int32(arc[1])
        lit = self.get_or_make_boolean_index(arc[2])
        model_ct.circuit.tails.append(tail)
        model_ct.circuit.heads.append(head)
        model_ct.circuit.literals.append(lit)
    return ct