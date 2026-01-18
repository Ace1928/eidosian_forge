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
def add_automaton(self, transition_variables: Sequence[VariableT], starting_state: IntegralT, final_states: Sequence[IntegralT], transition_triples: Sequence[Tuple[IntegralT, IntegralT, IntegralT]]) -> Constraint:
    """Adds an automaton constraint.

        An automaton constraint takes a list of variables (of size *n*), an initial
        state, a set of final states, and a set of transitions. A transition is a
        triplet (*tail*, *transition*, *head*), where *tail* and *head* are states,
        and *transition* is the label of an arc from *head* to *tail*,
        corresponding to the value of one variable in the list of variables.

        This automaton will be unrolled into a flow with *n* + 1 phases. Each phase
        contains the possible states of the automaton. The first state contains the
        initial state. The last phase contains the final states.

        Between two consecutive phases *i* and *i* + 1, the automaton creates a set
        of arcs. For each transition (*tail*, *transition*, *head*), it will add
        an arc from the state *tail* of phase *i* and the state *head* of phase
        *i* + 1. This arc is labeled by the value *transition* of the variables
        `variables[i]`. That is, this arc can only be selected if `variables[i]`
        is assigned the value *transition*.

        A feasible solution of this constraint is an assignment of variables such
        that, starting from the initial state in phase 0, there is a path labeled by
        the values of the variables that ends in one of the final states in the
        final phase.

        Args:
          transition_variables: A non-empty list of variables whose values
            correspond to the labels of the arcs traversed by the automaton.
          starting_state: The initial state of the automaton.
          final_states: A non-empty list of admissible final states.
          transition_triples: A list of transitions for the automaton, in the
            following format (current_state, variable_value, next_state).

        Returns:
          An instance of the `Constraint` class.

        Raises:
          ValueError: if `transition_variables`, `final_states`, or
            `transition_triples` are empty.
        """
    if not transition_variables:
        raise ValueError('add_automaton expects a non-empty transition_variables array')
    if not final_states:
        raise ValueError('add_automaton expects some final states')
    if not transition_triples:
        raise ValueError('add_automaton expects some transition triples')
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.automaton.vars.extend([self.get_or_make_index(x) for x in transition_variables])
    starting_state = cmh.assert_is_int64(starting_state)
    model_ct.automaton.starting_state = starting_state
    for v in final_states:
        v = cmh.assert_is_int64(v)
        model_ct.automaton.final_states.append(v)
    for t in transition_triples:
        if len(t) != 3:
            raise TypeError('Tuple ' + str(t) + ' has the wrong arity (!= 3)')
        tail = cmh.assert_is_int64(t[0])
        label = cmh.assert_is_int64(t[1])
        head = cmh.assert_is_int64(t[2])
        model_ct.automaton.transition_tail.append(tail)
        model_ct.automaton.transition_label.append(label)
        model_ct.automaton.transition_head.append(head)
    return ct