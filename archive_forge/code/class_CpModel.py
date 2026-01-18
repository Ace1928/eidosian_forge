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
class CpModel:
    """Methods for building a CP model.

    Methods beginning with:

    * ```New``` create integer, boolean, or interval variables.
    * ```add``` create new constraints and add them to the model.
    """

    def __init__(self):
        self.__model: cp_model_pb2.CpModelProto = cp_model_pb2.CpModelProto()
        self.__constant_map = {}

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        if not self.__model or not self.__model.name:
            return ''
        return self.__model.name

    @name.setter
    def name(self, name: str):
        """Sets the name of the model."""
        self.__model.name = name

    def new_int_var(self, lb: IntegralT, ub: IntegralT, name: str) -> IntVar:
        """Create an integer variable with domain [lb, ub].

        The CP-SAT solver is limited to integer variables. If you have fractional
        values, scale them up so that they become integers; if you have strings,
        encode them as integers.

        Args:
          lb: Lower bound for the variable.
          ub: Upper bound for the variable.
          name: The name of the variable.

        Returns:
          a variable whose domain is [lb, ub].
        """
        return IntVar(self.__model, Domain(lb, ub), name)

    def new_int_var_from_domain(self, domain: Domain, name: str) -> IntVar:
        """Create an integer variable from a domain.

        A domain is a set of integers specified by a collection of intervals.
        For example, `model.new_int_var_from_domain(cp_model.
             Domain.from_intervals([[1, 2], [4, 6]]), 'x')`

        Args:
          domain: An instance of the Domain class.
          name: The name of the variable.

        Returns:
            a variable whose domain is the given domain.
        """
        return IntVar(self.__model, domain, name)

    def new_bool_var(self, name: str) -> IntVar:
        """Creates a 0-1 variable with the given name."""
        return IntVar(self.__model, Domain(0, 1), name)

    def new_constant(self, value: IntegralT) -> IntVar:
        """Declares a constant integer."""
        return IntVar(self.__model, self.get_or_make_index_from_constant(value), None)

    def new_int_var_series(self, name: str, index: pd.Index, lower_bounds: Union[IntegralT, pd.Series], upper_bounds: Union[IntegralT, pd.Series]) -> pd.Series:
        """Creates a series of (scalar-valued) variables with the given name.

        Args:
          name (str): Required. The name of the variable set.
          index (pd.Index): Required. The index to use for the variable set.
          lower_bounds (Union[int, pd.Series]): A lower bound for variables in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.
          upper_bounds (Union[int, pd.Series]): An upper bound for variables in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.

        Returns:
          pd.Series: The variable set indexed by its corresponding dimensions.

        Raises:
          TypeError: if the `index` is invalid (e.g. a `DataFrame`).
          ValueError: if the `name` is not a valid identifier or already exists.
          ValueError: if the `lowerbound` is greater than the `upperbound`.
          ValueError: if the index of `lower_bound`, or `upper_bound` does not match
          the input index.
        """
        if not isinstance(index, pd.Index):
            raise TypeError('Non-index object is used as index')
        if not name.isidentifier():
            raise ValueError('name={} is not a valid identifier'.format(name))
        if isinstance(lower_bounds, numbers.Integral) and isinstance(upper_bounds, numbers.Integral) and (lower_bounds > upper_bounds):
            raise ValueError(f'lower_bound={lower_bounds} is greater than upper_bound={upper_bounds} for variable set={name}')
        lower_bounds = _convert_to_integral_series_and_validate_index(lower_bounds, index)
        upper_bounds = _convert_to_integral_series_and_validate_index(upper_bounds, index)
        return pd.Series(index=index, data=[IntVar(model=self.__model, name=f'{name}[{i}]', domain=Domain(lower_bounds[i], upper_bounds[i])) for i in index])

    def new_bool_var_series(self, name: str, index: pd.Index) -> pd.Series:
        """Creates a series of (scalar-valued) variables with the given name.

        Args:
          name (str): Required. The name of the variable set.
          index (pd.Index): Required. The index to use for the variable set.

        Returns:
          pd.Series: The variable set indexed by its corresponding dimensions.

        Raises:
          TypeError: if the `index` is invalid (e.g. a `DataFrame`).
          ValueError: if the `name` is not a valid identifier or already exists.
        """
        return self.new_int_var_series(name=name, index=index, lower_bounds=0, upper_bounds=1)

    def add_linear_constraint(self, linear_expr: LinearExprT, lb: IntegralT, ub: IntegralT) -> Constraint:
        """Adds the constraint: `lb <= linear_expr <= ub`."""
        return self.add_linear_expression_in_domain(linear_expr, Domain(lb, ub))

    def add_linear_expression_in_domain(self, linear_expr: LinearExprT, domain: Domain) -> Constraint:
        """Adds the constraint: `linear_expr` in `domain`."""
        if isinstance(linear_expr, LinearExpr):
            ct = Constraint(self)
            model_ct = self.__model.constraints[ct.index]
            coeffs_map, constant = linear_expr.get_integer_var_value_map()
            for t in coeffs_map.items():
                if not isinstance(t[0], IntVar):
                    raise TypeError('Wrong argument' + str(t))
                c = cmh.assert_is_int64(t[1])
                model_ct.linear.vars.append(t[0].index)
                model_ct.linear.coeffs.append(c)
            model_ct.linear.domain.extend([cmh.capped_subtraction(x, constant) for x in domain.flattened_intervals()])
            return ct
        if isinstance(linear_expr, numbers.Integral):
            if not domain.contains(int(linear_expr)):
                return self.add_bool_or([])
            else:
                return self.add_bool_and([])
        raise TypeError('not supported: CpModel.add_linear_expression_in_domain(' + str(linear_expr) + ' ' + str(domain) + ')')

    def add(self, ct: Union[BoundedLinearExpression, bool]) -> Constraint:
        """Adds a `BoundedLinearExpression` to the model.

        Args:
          ct: A [`BoundedLinearExpression`](#boundedlinearexpression).

        Returns:
          An instance of the `Constraint` class.
        """
        if isinstance(ct, BoundedLinearExpression):
            return self.add_linear_expression_in_domain(ct.expression(), Domain.from_flat_intervals(ct.bounds()))
        if ct and cmh.is_boolean(ct):
            return self.add_bool_or([True])
        if not ct and cmh.is_boolean(ct):
            return self.add_bool_or([])
        raise TypeError('not supported: CpModel.add(' + str(ct) + ')')

    @overload
    def add_all_different(self, expressions: Iterable[LinearExprT]) -> Constraint:
        ...

    @overload
    def add_all_different(self, *expressions: LinearExprT) -> Constraint:
        ...

    def add_all_different(self, *expressions):
        """Adds AllDifferent(expressions).

        This constraint forces all expressions to have different values.

        Args:
          *expressions: simple expressions of the form a * var + constant.

        Returns:
          An instance of the `Constraint` class.
        """
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        expanded = expand_generator_or_tuple(expressions)
        model_ct.all_diff.exprs.extend((self.parse_linear_expression(x) for x in expanded))
        return ct

    def add_element(self, index: VariableT, variables: Sequence[VariableT], target: VariableT) -> Constraint:
        """Adds the element constraint: `variables[index] == target`.

        Args:
          index: The index of the variable that's being constrained.
          variables: A list of variables.
          target: The value that the variable must be equal to.

        Returns:
          An instance of the `Constraint` class.
        """
        if not variables:
            raise ValueError('add_element expects a non-empty variables array')
        if isinstance(index, numbers.Integral):
            return self.add(list(variables)[int(index)] == target)
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.element.index = self.get_or_make_index(index)
        model_ct.element.vars.extend([self.get_or_make_index(x) for x in variables])
        model_ct.element.target = self.get_or_make_index(target)
        return ct

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

    def add_multiple_circuit(self, arcs: Sequence[ArcT]) -> Constraint:
        """Adds a multiple circuit constraint, aka the 'VRP' constraint.

        The direct graph where arc #i (from tails[i] to head[i]) is present iff
        literals[i] is true must satisfy this set of properties:
        - #incoming arcs == 1 except for node 0.
        - #outgoing arcs == 1 except for node 0.
        - for node zero, #incoming arcs == #outgoing arcs.
        - There are no duplicate arcs.
        - Self-arcs are allowed except for node 0.
        - There is no cycle in this graph, except through node 0.

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
            raise ValueError('add_multiple_circuit expects a non-empty array of arcs')
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        for arc in arcs:
            tail = cmh.assert_is_int32(arc[0])
            head = cmh.assert_is_int32(arc[1])
            lit = self.get_or_make_boolean_index(arc[2])
            model_ct.routes.tails.append(tail)
            model_ct.routes.heads.append(head)
            model_ct.routes.literals.append(lit)
        return ct

    def add_allowed_assignments(self, variables: Sequence[VariableT], tuples_list: Iterable[Sequence[IntegralT]]) -> Constraint:
        """Adds AllowedAssignments(variables, tuples_list).

        An AllowedAssignments constraint is a constraint on an array of variables,
        which requires that when all variables are assigned values, the resulting
        array equals one of the  tuples in `tuple_list`.

        Args:
          variables: A list of variables.
          tuples_list: A list of admissible tuples. Each tuple must have the same
            length as the variables, and the ith value of a tuple corresponds to the
            ith variable.

        Returns:
          An instance of the `Constraint` class.

        Raises:
          TypeError: If a tuple does not have the same size as the list of
              variables.
          ValueError: If the array of variables is empty.
        """
        if not variables:
            raise ValueError('add_allowed_assignments expects a non-empty variables array')
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.table.vars.extend([self.get_or_make_index(x) for x in variables])
        arity = len(variables)
        for t in tuples_list:
            if len(t) != arity:
                raise TypeError('Tuple ' + str(t) + ' has the wrong arity')
            ar = []
            for v in t:
                ar.append(cmh.assert_is_int64(v))
            model_ct.table.values.extend(ar)
        return ct

    def add_forbidden_assignments(self, variables: Sequence[VariableT], tuples_list: Iterable[Sequence[IntegralT]]) -> Constraint:
        """Adds add_forbidden_assignments(variables, [tuples_list]).

        A ForbiddenAssignments constraint is a constraint on an array of variables
        where the list of impossible combinations is provided in the tuples list.

        Args:
          variables: A list of variables.
          tuples_list: A list of forbidden tuples. Each tuple must have the same
            length as the variables, and the *i*th value of a tuple corresponds to
            the *i*th variable.

        Returns:
          An instance of the `Constraint` class.

        Raises:
          TypeError: If a tuple does not have the same size as the list of
                     variables.
          ValueError: If the array of variables is empty.
        """
        if not variables:
            raise ValueError('add_forbidden_assignments expects a non-empty variables array')
        index = len(self.__model.constraints)
        ct = self.add_allowed_assignments(variables, tuples_list)
        self.__model.constraints[index].table.negated = True
        return ct

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

    def add_inverse(self, variables: Sequence[VariableT], inverse_variables: Sequence[VariableT]) -> Constraint:
        """Adds Inverse(variables, inverse_variables).

        An inverse constraint enforces that if `variables[i]` is assigned a value
        `j`, then `inverse_variables[j]` is assigned a value `i`. And vice versa.

        Args:
          variables: An array of integer variables.
          inverse_variables: An array of integer variables.

        Returns:
          An instance of the `Constraint` class.

        Raises:
          TypeError: if variables and inverse_variables have different lengths, or
              if they are empty.
        """
        if not variables or not inverse_variables:
            raise TypeError('The Inverse constraint does not accept empty arrays')
        if len(variables) != len(inverse_variables):
            raise TypeError('In the inverse constraint, the two array variables and inverse_variables must have the same length.')
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.inverse.f_direct.extend([self.get_or_make_index(x) for x in variables])
        model_ct.inverse.f_inverse.extend([self.get_or_make_index(x) for x in inverse_variables])
        return ct

    def add_reservoir_constraint(self, times: Iterable[LinearExprT], level_changes: Iterable[LinearExprT], min_level: int, max_level: int) -> Constraint:
        """Adds Reservoir(times, level_changes, min_level, max_level).

        Maintains a reservoir level within bounds. The water level starts at 0, and
        at any time, it must be between min_level and max_level.

        If the affine expression `times[i]` is assigned a value t, then the current
        level changes by `level_changes[i]`, which is constant, at time t.

         Note that min level must be <= 0, and the max level must be >= 0. Please
         use fixed level_changes to simulate initial state.

         Therefore, at any time:
             sum(level_changes[i] if times[i] <= t) in [min_level, max_level]

        Args:
          times: A list of 1-var affine expressions (a * x + b) which specify the
            time of the filling or emptying the reservoir.
          level_changes: A list of integer values that specifies the amount of the
            emptying or filling. Currently, variable demands are not supported.
          min_level: At any time, the level of the reservoir must be greater or
            equal than the min level.
          max_level: At any time, the level of the reservoir must be less or equal
            than the max level.

        Returns:
          An instance of the `Constraint` class.

        Raises:
          ValueError: if max_level < min_level.

          ValueError: if max_level < 0.

          ValueError: if min_level > 0
        """
        if max_level < min_level:
            raise ValueError('Reservoir constraint must have a max_level >= min_level')
        if max_level < 0:
            raise ValueError('Reservoir constraint must have a max_level >= 0')
        if min_level > 0:
            raise ValueError('Reservoir constraint must have a min_level <= 0')
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.reservoir.time_exprs.extend([self.parse_linear_expression(x) for x in times])
        model_ct.reservoir.level_changes.extend([self.parse_linear_expression(x) for x in level_changes])
        model_ct.reservoir.min_level = min_level
        model_ct.reservoir.max_level = max_level
        return ct

    def add_reservoir_constraint_with_active(self, times: Iterable[LinearExprT], level_changes: Iterable[LinearExprT], actives: Iterable[LiteralT], min_level: int, max_level: int) -> Constraint:
        """Adds Reservoir(times, level_changes, actives, min_level, max_level).

        Maintains a reservoir level within bounds. The water level starts at 0, and
        at any time, it must be between min_level and max_level.

        If the variable `times[i]` is assigned a value t, and `actives[i]` is
        `True`, then the current level changes by `level_changes[i]`, which is
        constant,
        at time t.

         Note that min level must be <= 0, and the max level must be >= 0. Please
         use fixed level_changes to simulate initial state.

         Therefore, at any time:
             sum(level_changes[i] * actives[i] if times[i] <= t) in [min_level,
             max_level]


        The array of boolean variables 'actives', if defined, indicates which
        actions are actually performed.

        Args:
          times: A list of 1-var affine expressions (a * x + b) which specify the
            time of the filling or emptying the reservoir.
          level_changes: A list of integer values that specifies the amount of the
            emptying or filling. Currently, variable demands are not supported.
          actives: a list of boolean variables. They indicates if the
            emptying/refilling events actually take place.
          min_level: At any time, the level of the reservoir must be greater or
            equal than the min level.
          max_level: At any time, the level of the reservoir must be less or equal
            than the max level.

        Returns:
          An instance of the `Constraint` class.

        Raises:
          ValueError: if max_level < min_level.

          ValueError: if max_level < 0.

          ValueError: if min_level > 0
        """
        if max_level < min_level:
            raise ValueError('Reservoir constraint must have a max_level >= min_level')
        if max_level < 0:
            raise ValueError('Reservoir constraint must have a max_level >= 0')
        if min_level > 0:
            raise ValueError('Reservoir constraint must have a min_level <= 0')
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.reservoir.time_exprs.extend([self.parse_linear_expression(x) for x in times])
        model_ct.reservoir.level_changes.extend([self.parse_linear_expression(x) for x in level_changes])
        model_ct.reservoir.active_literals.extend([self.get_or_make_boolean_index(x) for x in actives])
        model_ct.reservoir.min_level = min_level
        model_ct.reservoir.max_level = max_level
        return ct

    def add_map_domain(self, var: IntVar, bool_var_array: Iterable[IntVar], offset: IntegralT=0):
        """Adds `var == i + offset <=> bool_var_array[i] == true for all i`."""
        for i, bool_var in enumerate(bool_var_array):
            b_index = bool_var.index
            var_index = var.index
            model_ct = self.__model.constraints.add()
            model_ct.linear.vars.append(var_index)
            model_ct.linear.coeffs.append(1)
            model_ct.linear.domain.extend([offset + i, offset + i])
            model_ct.enforcement_literal.append(b_index)
            model_ct = self.__model.constraints.add()
            model_ct.linear.vars.append(var_index)
            model_ct.linear.coeffs.append(1)
            model_ct.enforcement_literal.append(-b_index - 1)
            if offset + i - 1 >= INT_MIN:
                model_ct.linear.domain.extend([INT_MIN, offset + i - 1])
            if offset + i + 1 <= INT_MAX:
                model_ct.linear.domain.extend([offset + i + 1, INT_MAX])

    def add_implication(self, a: LiteralT, b: LiteralT) -> Constraint:
        """Adds `a => b` (`a` implies `b`)."""
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.bool_or.literals.append(self.get_or_make_boolean_index(b))
        model_ct.enforcement_literal.append(self.get_or_make_boolean_index(a))
        return ct

    @overload
    def add_bool_or(self, literals: Iterable[LiteralT]) -> Constraint:
        ...

    @overload
    def add_bool_or(self, *literals: LiteralT) -> Constraint:
        ...

    def add_bool_or(self, *literals):
        """Adds `Or(literals) == true`: sum(literals) >= 1."""
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.bool_or.literals.extend([self.get_or_make_boolean_index(x) for x in expand_generator_or_tuple(literals)])
        return ct

    @overload
    def add_at_least_one(self, literals: Iterable[LiteralT]) -> Constraint:
        ...

    @overload
    def add_at_least_one(self, *literals: LiteralT) -> Constraint:
        ...

    def add_at_least_one(self, *literals):
        """Same as `add_bool_or`: `sum(literals) >= 1`."""
        return self.add_bool_or(*literals)

    @overload
    def add_at_most_one(self, literals: Iterable[LiteralT]) -> Constraint:
        ...

    @overload
    def add_at_most_one(self, *literals: LiteralT) -> Constraint:
        ...

    def add_at_most_one(self, *literals):
        """Adds `AtMostOne(literals)`: `sum(literals) <= 1`."""
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.at_most_one.literals.extend([self.get_or_make_boolean_index(x) for x in expand_generator_or_tuple(literals)])
        return ct

    @overload
    def add_exactly_one(self, literals: Iterable[LiteralT]) -> Constraint:
        ...

    @overload
    def add_exactly_one(self, *literals: LiteralT) -> Constraint:
        ...

    def add_exactly_one(self, *literals):
        """Adds `ExactlyOne(literals)`: `sum(literals) == 1`."""
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.exactly_one.literals.extend([self.get_or_make_boolean_index(x) for x in expand_generator_or_tuple(literals)])
        return ct

    @overload
    def add_bool_and(self, literals: Iterable[LiteralT]) -> Constraint:
        ...

    @overload
    def add_bool_and(self, *literals: LiteralT) -> Constraint:
        ...

    def add_bool_and(self, *literals):
        """Adds `And(literals) == true`."""
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.bool_and.literals.extend([self.get_or_make_boolean_index(x) for x in expand_generator_or_tuple(literals)])
        return ct

    @overload
    def add_bool_xor(self, literals: Iterable[LiteralT]) -> Constraint:
        ...

    @overload
    def add_bool_xor(self, *literals: LiteralT) -> Constraint:
        ...

    def add_bool_xor(self, *literals):
        """Adds `XOr(literals) == true`.

        In contrast to add_bool_or and add_bool_and, it does not support
            .only_enforce_if().

        Args:
          *literals: the list of literals in the constraint.

        Returns:
          An `Constraint` object.
        """
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.bool_xor.literals.extend([self.get_or_make_boolean_index(x) for x in expand_generator_or_tuple(literals)])
        return ct

    def add_min_equality(self, target: LinearExprT, exprs: Iterable[LinearExprT]) -> Constraint:
        """Adds `target == Min(exprs)`."""
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.lin_max.exprs.extend([self.parse_linear_expression(x, True) for x in exprs])
        model_ct.lin_max.target.CopyFrom(self.parse_linear_expression(target, True))
        return ct

    def add_max_equality(self, target: LinearExprT, exprs: Iterable[LinearExprT]) -> Constraint:
        """Adds `target == Max(exprs)`."""
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.lin_max.exprs.extend([self.parse_linear_expression(x) for x in exprs])
        model_ct.lin_max.target.CopyFrom(self.parse_linear_expression(target))
        return ct

    def add_division_equality(self, target: LinearExprT, num: LinearExprT, denom: LinearExprT) -> Constraint:
        """Adds `target == num // denom` (integer division rounded towards 0)."""
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.int_div.exprs.append(self.parse_linear_expression(num))
        model_ct.int_div.exprs.append(self.parse_linear_expression(denom))
        model_ct.int_div.target.CopyFrom(self.parse_linear_expression(target))
        return ct

    def add_abs_equality(self, target: LinearExprT, expr: LinearExprT) -> Constraint:
        """Adds `target == Abs(expr)`."""
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.lin_max.exprs.append(self.parse_linear_expression(expr))
        model_ct.lin_max.exprs.append(self.parse_linear_expression(expr, True))
        model_ct.lin_max.target.CopyFrom(self.parse_linear_expression(target))
        return ct

    def add_modulo_equality(self, target: LinearExprT, expr: LinearExprT, mod: LinearExprT) -> Constraint:
        """Adds `target = expr % mod`."""
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.int_mod.exprs.append(self.parse_linear_expression(expr))
        model_ct.int_mod.exprs.append(self.parse_linear_expression(mod))
        model_ct.int_mod.target.CopyFrom(self.parse_linear_expression(target))
        return ct

    def add_multiplication_equality(self, target: LinearExprT, *expressions: Union[Iterable[LinearExprT], LinearExprT]) -> Constraint:
        """Adds `target == expressions[0] * .. * expressions[n]`."""
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.int_prod.exprs.extend([self.parse_linear_expression(expr) for expr in expand_generator_or_tuple(expressions)])
        model_ct.int_prod.target.CopyFrom(self.parse_linear_expression(target))
        return ct

    def new_interval_var(self, start: LinearExprT, size: LinearExprT, end: LinearExprT, name: str) -> IntervalVar:
        """Creates an interval variable from start, size, and end.

        An interval variable is a constraint, that is itself used in other
        constraints like NoOverlap.

        Internally, it ensures that `start + size == end`.

        Args:
          start: The start of the interval. It must be of the form a * var + b.
          size: The size of the interval. It must be of the form a * var + b.
          end: The end of the interval. It must be of the form a * var + b.
          name: The name of the interval variable.

        Returns:
          An `IntervalVar` object.
        """
        lin = self.add(start + size == end)
        if name:
            lin.with_name('lin_' + name)
        start_expr = self.parse_linear_expression(start)
        size_expr = self.parse_linear_expression(size)
        end_expr = self.parse_linear_expression(end)
        if len(start_expr.vars) > 1:
            raise TypeError('cp_model.new_interval_var: start must be 1-var affine or constant.')
        if len(size_expr.vars) > 1:
            raise TypeError('cp_model.new_interval_var: size must be 1-var affine or constant.')
        if len(end_expr.vars) > 1:
            raise TypeError('cp_model.new_interval_var: end must be 1-var affine or constant.')
        return IntervalVar(self.__model, start_expr, size_expr, end_expr, None, name)

    def new_interval_var_series(self, name: str, index: pd.Index, starts: Union[LinearExprT, pd.Series], sizes: Union[LinearExprT, pd.Series], ends: Union[LinearExprT, pd.Series]) -> pd.Series:
        """Creates a series of interval variables with the given name.

        Args:
          name (str): Required. The name of the variable set.
          index (pd.Index): Required. The index to use for the variable set.
          starts (Union[LinearExprT, pd.Series]): The start of each interval in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.
          sizes (Union[LinearExprT, pd.Series]): The size of each interval in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.
          ends (Union[LinearExprT, pd.Series]): The ends of each interval in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.

        Returns:
          pd.Series: The interval variable set indexed by its corresponding
          dimensions.

        Raises:
          TypeError: if the `index` is invalid (e.g. a `DataFrame`).
          ValueError: if the `name` is not a valid identifier or already exists.
          ValueError: if the all the indexes do not match.
        """
        if not isinstance(index, pd.Index):
            raise TypeError('Non-index object is used as index')
        if not name.isidentifier():
            raise ValueError('name={} is not a valid identifier'.format(name))
        starts = _convert_to_linear_expr_series_and_validate_index(starts, index)
        sizes = _convert_to_linear_expr_series_and_validate_index(sizes, index)
        ends = _convert_to_linear_expr_series_and_validate_index(ends, index)
        interval_array = []
        for i in index:
            interval_array.append(self.new_interval_var(start=starts[i], size=sizes[i], end=ends[i], name=f'{name}[{i}]'))
        return pd.Series(index=index, data=interval_array)

    def new_fixed_size_interval_var(self, start: LinearExprT, size: IntegralT, name: str) -> IntervalVar:
        """Creates an interval variable from start, and a fixed size.

        An interval variable is a constraint, that is itself used in other
        constraints like NoOverlap.

        Args:
          start: The start of the interval. It must be of the form a * var + b.
          size: The size of the interval. It must be an integer value.
          name: The name of the interval variable.

        Returns:
          An `IntervalVar` object.
        """
        size = cmh.assert_is_int64(size)
        start_expr = self.parse_linear_expression(start)
        size_expr = self.parse_linear_expression(size)
        end_expr = self.parse_linear_expression(start + size)
        if len(start_expr.vars) > 1:
            raise TypeError('cp_model.new_interval_var: start must be affine or constant.')
        return IntervalVar(self.__model, start_expr, size_expr, end_expr, None, name)

    def new_fixed_size_interval_var_series(self, name: str, index: pd.Index, starts: Union[LinearExprT, pd.Series], sizes: Union[IntegralT, pd.Series]) -> pd.Series:
        """Creates a series of interval variables with the given name.

        Args:
          name (str): Required. The name of the variable set.
          index (pd.Index): Required. The index to use for the variable set.
          starts (Union[LinearExprT, pd.Series]): The start of each interval in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.
          sizes (Union[IntegralT, pd.Series]): The fixed size of each interval in
            the set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.

        Returns:
          pd.Series: The interval variable set indexed by its corresponding
          dimensions.

        Raises:
          TypeError: if the `index` is invalid (e.g. a `DataFrame`).
          ValueError: if the `name` is not a valid identifier or already exists.
          ValueError: if the all the indexes do not match.
        """
        if not isinstance(index, pd.Index):
            raise TypeError('Non-index object is used as index')
        if not name.isidentifier():
            raise ValueError('name={} is not a valid identifier'.format(name))
        starts = _convert_to_linear_expr_series_and_validate_index(starts, index)
        sizes = _convert_to_integral_series_and_validate_index(sizes, index)
        interval_array = []
        for i in index:
            interval_array.append(self.new_fixed_size_interval_var(start=starts[i], size=sizes[i], name=f'{name}[{i}]'))
        return pd.Series(index=index, data=interval_array)

    def new_optional_interval_var(self, start: LinearExprT, size: LinearExprT, end: LinearExprT, is_present: LiteralT, name: str) -> IntervalVar:
        """Creates an optional interval var from start, size, end, and is_present.

        An optional interval variable is a constraint, that is itself used in other
        constraints like NoOverlap. This constraint is protected by a presence
        literal that indicates if it is active or not.

        Internally, it ensures that `is_present` implies `start + size ==
        end`.

        Args:
          start: The start of the interval. It must be of the form a * var + b.
          size: The size of the interval. It must be of the form a * var + b.
          end: The end of the interval. It must be of the form a * var + b.
          is_present: A literal that indicates if the interval is active or not. A
            inactive interval is simply ignored by all constraints.
          name: The name of the interval variable.

        Returns:
          An `IntervalVar` object.
        """
        lin = self.add(start + size == end).only_enforce_if(is_present)
        if name:
            lin.with_name('lin_opt_' + name)
        is_present_index = self.get_or_make_boolean_index(is_present)
        start_expr = self.parse_linear_expression(start)
        size_expr = self.parse_linear_expression(size)
        end_expr = self.parse_linear_expression(end)
        if len(start_expr.vars) > 1:
            raise TypeError('cp_model.new_interval_var: start must be affine or constant.')
        if len(size_expr.vars) > 1:
            raise TypeError('cp_model.new_interval_var: size must be affine or constant.')
        if len(end_expr.vars) > 1:
            raise TypeError('cp_model.new_interval_var: end must be affine or constant.')
        return IntervalVar(self.__model, start_expr, size_expr, end_expr, is_present_index, name)

    def new_optional_interval_var_series(self, name: str, index: pd.Index, starts: Union[LinearExprT, pd.Series], sizes: Union[LinearExprT, pd.Series], ends: Union[LinearExprT, pd.Series], are_present: Union[LiteralT, pd.Series]) -> pd.Series:
        """Creates a series of interval variables with the given name.

        Args:
          name (str): Required. The name of the variable set.
          index (pd.Index): Required. The index to use for the variable set.
          starts (Union[LinearExprT, pd.Series]): The start of each interval in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.
          sizes (Union[LinearExprT, pd.Series]): The size of each interval in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.
          ends (Union[LinearExprT, pd.Series]): The ends of each interval in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.
          are_present (Union[LiteralT, pd.Series]): The performed literal of each
            interval in the set. If a `pd.Series` is passed in, it will be based on
            the corresponding values of the pd.Series.

        Returns:
          pd.Series: The interval variable set indexed by its corresponding
          dimensions.

        Raises:
          TypeError: if the `index` is invalid (e.g. a `DataFrame`).
          ValueError: if the `name` is not a valid identifier or already exists.
          ValueError: if the all the indexes do not match.
        """
        if not isinstance(index, pd.Index):
            raise TypeError('Non-index object is used as index')
        if not name.isidentifier():
            raise ValueError('name={} is not a valid identifier'.format(name))
        starts = _convert_to_linear_expr_series_and_validate_index(starts, index)
        sizes = _convert_to_linear_expr_series_and_validate_index(sizes, index)
        ends = _convert_to_linear_expr_series_and_validate_index(ends, index)
        are_present = _convert_to_literal_series_and_validate_index(are_present, index)
        interval_array = []
        for i in index:
            interval_array.append(self.new_optional_interval_var(start=starts[i], size=sizes[i], end=ends[i], is_present=are_present[i], name=f'{name}[{i}]'))
        return pd.Series(index=index, data=interval_array)

    def new_optional_fixed_size_interval_var(self, start: LinearExprT, size: IntegralT, is_present: LiteralT, name: str) -> IntervalVar:
        """Creates an interval variable from start, and a fixed size.

        An interval variable is a constraint, that is itself used in other
        constraints like NoOverlap.

        Args:
          start: The start of the interval. It must be of the form a * var + b.
          size: The size of the interval. It must be an integer value.
          is_present: A literal that indicates if the interval is active or not. A
            inactive interval is simply ignored by all constraints.
          name: The name of the interval variable.

        Returns:
          An `IntervalVar` object.
        """
        size = cmh.assert_is_int64(size)
        start_expr = self.parse_linear_expression(start)
        size_expr = self.parse_linear_expression(size)
        end_expr = self.parse_linear_expression(start + size)
        if len(start_expr.vars) > 1:
            raise TypeError('cp_model.new_interval_var: start must be affine or constant.')
        is_present_index = self.get_or_make_boolean_index(is_present)
        return IntervalVar(self.__model, start_expr, size_expr, end_expr, is_present_index, name)

    def new_optional_fixed_size_interval_var_series(self, name: str, index: pd.Index, starts: Union[LinearExprT, pd.Series], sizes: Union[IntegralT, pd.Series], are_present: Union[LiteralT, pd.Series]) -> pd.Series:
        """Creates a series of interval variables with the given name.

        Args:
          name (str): Required. The name of the variable set.
          index (pd.Index): Required. The index to use for the variable set.
          starts (Union[LinearExprT, pd.Series]): The start of each interval in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.
          sizes (Union[IntegralT, pd.Series]): The fixed size of each interval in
            the set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.
          are_present (Union[LiteralT, pd.Series]): The performed literal of each
            interval in the set. If a `pd.Series` is passed in, it will be based on
            the corresponding values of the pd.Series.

        Returns:
          pd.Series: The interval variable set indexed by its corresponding
          dimensions.

        Raises:
          TypeError: if the `index` is invalid (e.g. a `DataFrame`).
          ValueError: if the `name` is not a valid identifier or already exists.
          ValueError: if the all the indexes do not match.
        """
        if not isinstance(index, pd.Index):
            raise TypeError('Non-index object is used as index')
        if not name.isidentifier():
            raise ValueError('name={} is not a valid identifier'.format(name))
        starts = _convert_to_linear_expr_series_and_validate_index(starts, index)
        sizes = _convert_to_integral_series_and_validate_index(sizes, index)
        are_present = _convert_to_literal_series_and_validate_index(are_present, index)
        interval_array = []
        for i in index:
            interval_array.append(self.new_optional_fixed_size_interval_var(start=starts[i], size=sizes[i], is_present=are_present[i], name=f'{name}[{i}]'))
        return pd.Series(index=index, data=interval_array)

    def add_no_overlap(self, interval_vars: Iterable[IntervalVar]) -> Constraint:
        """Adds NoOverlap(interval_vars).

        A NoOverlap constraint ensures that all present intervals do not overlap
        in time.

        Args:
          interval_vars: The list of interval variables to constrain.

        Returns:
          An instance of the `Constraint` class.
        """
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.no_overlap.intervals.extend([self.get_interval_index(x) for x in interval_vars])
        return ct

    def add_no_overlap_2d(self, x_intervals: Iterable[IntervalVar], y_intervals: Iterable[IntervalVar]) -> Constraint:
        """Adds NoOverlap2D(x_intervals, y_intervals).

        A NoOverlap2D constraint ensures that all present rectangles do not overlap
        on a plane. Each rectangle is aligned with the X and Y axis, and is defined
        by two intervals which represent its projection onto the X and Y axis.

        Furthermore, one box is optional if at least one of the x or y interval is
        optional.

        Args:
          x_intervals: The X coordinates of the rectangles.
          y_intervals: The Y coordinates of the rectangles.

        Returns:
          An instance of the `Constraint` class.
        """
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        model_ct.no_overlap_2d.x_intervals.extend([self.get_interval_index(x) for x in x_intervals])
        model_ct.no_overlap_2d.y_intervals.extend([self.get_interval_index(x) for x in y_intervals])
        return ct

    def add_cumulative(self, intervals: Iterable[IntervalVar], demands: Iterable[LinearExprT], capacity: LinearExprT) -> Constraint:
        """Adds Cumulative(intervals, demands, capacity).

        This constraint enforces that:

            for all t:
              sum(demands[i]
                if (start(intervals[i]) <= t < end(intervals[i])) and
                (intervals[i] is present)) <= capacity

        Args:
          intervals: The list of intervals.
          demands: The list of demands for each interval. Each demand must be >= 0.
            Each demand can be a 1-var affine expression (a * x + b).
          capacity: The maximum capacity of the cumulative constraint. It can be a
            1-var affine expression (a * x + b).

        Returns:
          An instance of the `Constraint` class.
        """
        cumulative = Constraint(self)
        model_ct = self.__model.constraints[cumulative.index]
        model_ct.cumulative.intervals.extend([self.get_interval_index(x) for x in intervals])
        for d in demands:
            model_ct.cumulative.demands.append(self.parse_linear_expression(d))
        model_ct.cumulative.capacity.CopyFrom(self.parse_linear_expression(capacity))
        return cumulative

    def clone(self) -> 'CpModel':
        """Reset the model, and creates a new one from a CpModelProto instance."""
        clone = CpModel()
        clone.proto.CopyFrom(self.proto)
        clone.rebuild_constant_map()
        return clone

    def rebuild_constant_map(self):
        """Internal method used during model cloning."""
        for i, var in enumerate(self.__model.variables):
            if len(var.domain) == 2 and var.domain[0] == var.domain[1]:
                self.__constant_map[var.domain[0]] = i

    def get_bool_var_from_proto_index(self, index: int) -> IntVar:
        """Returns an already created Boolean variable from its index."""
        if index < 0 or index >= len(self.__model.variables):
            raise ValueError(f'get_bool_var_from_proto_index: out of bound index {index}')
        var = self.__model.variables[index]
        if len(var.domain) != 2 or var.domain[0] < 0 or var.domain[1] > 1:
            raise ValueError(f'get_bool_var_from_proto_index: index {index} does not reference' + ' a Boolean variable')
        return IntVar(self.__model, index, None)

    def get_int_var_from_proto_index(self, index: int) -> IntVar:
        """Returns an already created integer variable from its index."""
        if index < 0 or index >= len(self.__model.variables):
            raise ValueError(f'get_int_var_from_proto_index: out of bound index {index}')
        return IntVar(self.__model, index, None)

    def get_interval_var_from_proto_index(self, index: int) -> IntervalVar:
        """Returns an already created interval variable from its index."""
        if index < 0 or index >= len(self.__model.constraints):
            raise ValueError(f'get_interval_var_from_proto_index: out of bound index {index}')
        ct = self.__model.constraints[index]
        if not ct.HasField('interval'):
            raise ValueError(f'get_interval_var_from_proto_index: index {index} does not reference an' + ' interval variable')
        return IntervalVar(self.__model, index, None, None, None, None)

    def __str__(self):
        return str(self.__model)

    @property
    def proto(self) -> cp_model_pb2.CpModelProto:
        """Returns the underlying CpModelProto."""
        return self.__model

    def negated(self, index: int) -> int:
        return -index - 1

    def get_or_make_index(self, arg: VariableT) -> int:
        """Returns the index of a variable, its negation, or a number."""
        if isinstance(arg, IntVar):
            return arg.index
        if isinstance(arg, _ProductCst) and isinstance(arg.expression(), IntVar) and (arg.coefficient() == -1):
            return -arg.expression().index - 1
        if isinstance(arg, numbers.Integral):
            arg = cmh.assert_is_int64(arg)
            return self.get_or_make_index_from_constant(arg)
        raise TypeError('NotSupported: model.get_or_make_index(' + str(arg) + ')')

    def get_or_make_boolean_index(self, arg: LiteralT) -> int:
        """Returns an index from a boolean expression."""
        if isinstance(arg, IntVar):
            self.assert_is_boolean_variable(arg)
            return arg.index
        if isinstance(arg, _NotBooleanVariable):
            self.assert_is_boolean_variable(arg.negated())
            return arg.index
        if isinstance(arg, numbers.Integral):
            arg = cmh.assert_is_zero_or_one(arg)
            return self.get_or_make_index_from_constant(arg)
        if cmh.is_boolean(arg):
            return self.get_or_make_index_from_constant(int(arg))
        raise TypeError(f'not supported: model.get_or_make_boolean_index({arg})')

    def get_interval_index(self, arg: IntervalVar) -> int:
        if not isinstance(arg, IntervalVar):
            raise TypeError('NotSupported: model.get_interval_index(%s)' % arg)
        return arg.index

    def get_or_make_index_from_constant(self, value: IntegralT) -> int:
        if value in self.__constant_map:
            return self.__constant_map[value]
        index = len(self.__model.variables)
        self.__model.variables.add(domain=[value, value])
        self.__constant_map[value] = index
        return index

    def var_index_to_var_proto(self, var_index: int) -> cp_model_pb2.IntegerVariableProto:
        if var_index >= 0:
            return self.__model.variables[var_index]
        else:
            return self.__model.variables[-var_index - 1]

    def parse_linear_expression(self, linear_expr: LinearExprT, negate: bool=False) -> cp_model_pb2.LinearExpressionProto:
        """Returns a LinearExpressionProto built from a LinearExpr instance."""
        result: cp_model_pb2.LinearExpressionProto = cp_model_pb2.LinearExpressionProto()
        mult = -1 if negate else 1
        if isinstance(linear_expr, numbers.Integral):
            result.offset = int(linear_expr) * mult
            return result
        if isinstance(linear_expr, IntVar):
            result.vars.append(self.get_or_make_index(linear_expr))
            result.coeffs.append(mult)
            return result
        coeffs_map, constant = cast(LinearExpr, linear_expr).get_integer_var_value_map()
        result.offset = constant * mult
        for t in coeffs_map.items():
            if not isinstance(t[0], IntVar):
                raise TypeError('Wrong argument' + str(t))
            c = cmh.assert_is_int64(t[1])
            result.vars.append(t[0].index)
            result.coeffs.append(c * mult)
        return result

    def _set_objective(self, obj: ObjLinearExprT, minimize: bool):
        """Sets the objective of the model."""
        self.clear_objective()
        if isinstance(obj, IntVar):
            self.__model.objective.coeffs.append(1)
            self.__model.objective.offset = 0
            if minimize:
                self.__model.objective.vars.append(obj.index)
                self.__model.objective.scaling_factor = 1
            else:
                self.__model.objective.vars.append(self.negated(obj.index))
                self.__model.objective.scaling_factor = -1
        elif isinstance(obj, LinearExpr):
            coeffs_map, constant, is_integer = obj.get_float_var_value_map()
            if is_integer:
                if minimize:
                    self.__model.objective.scaling_factor = 1
                    self.__model.objective.offset = constant
                else:
                    self.__model.objective.scaling_factor = -1
                    self.__model.objective.offset = -constant
                for v, c in coeffs_map.items():
                    self.__model.objective.coeffs.append(c)
                    if minimize:
                        self.__model.objective.vars.append(v.index)
                    else:
                        self.__model.objective.vars.append(self.negated(v.index))
            else:
                self.__model.floating_point_objective.maximize = not minimize
                self.__model.floating_point_objective.offset = constant
                for v, c in coeffs_map.items():
                    self.__model.floating_point_objective.coeffs.append(c)
                    self.__model.floating_point_objective.vars.append(v.index)
        elif isinstance(obj, numbers.Integral):
            self.__model.objective.offset = int(obj)
            self.__model.objective.scaling_factor = 1
        else:
            raise TypeError('TypeError: ' + str(obj) + ' is not a valid objective')

    def minimize(self, obj: ObjLinearExprT):
        """Sets the objective of the model to minimize(obj)."""
        self._set_objective(obj, minimize=True)

    def maximize(self, obj: ObjLinearExprT):
        """Sets the objective of the model to maximize(obj)."""
        self._set_objective(obj, minimize=False)

    def has_objective(self) -> bool:
        return self.__model.HasField('objective') or self.__model.HasField('floating_point_objective')

    def clear_objective(self):
        self.__model.ClearField('objective')
        self.__model.ClearField('floating_point_objective')

    def add_decision_strategy(self, variables: Sequence[IntVar], var_strategy: cp_model_pb2.DecisionStrategyProto.VariableSelectionStrategy, domain_strategy: cp_model_pb2.DecisionStrategyProto.DomainReductionStrategy) -> None:
        """Adds a search strategy to the model.

        Args:
          variables: a list of variables this strategy will assign.
          var_strategy: heuristic to choose the next variable to assign.
          domain_strategy: heuristic to reduce the domain of the selected variable.
            Currently, this is advanced code: the union of all strategies added to
            the model must be complete, i.e. instantiates all variables. Otherwise,
            solve() will fail.
        """
        strategy = self.__model.search_strategy.add()
        for v in variables:
            expr = strategy.exprs.add()
            if v.index >= 0:
                expr.vars.append(v.index)
                expr.coeffs.append(1)
            else:
                expr.vars.append(self.negated(v.index))
                expr.coeffs.append(-1)
                expr.offset = 1
        strategy.variable_selection_strategy = var_strategy
        strategy.domain_reduction_strategy = domain_strategy

    def model_stats(self) -> str:
        """Returns a string containing some model statistics."""
        return swig_helper.CpSatHelper.model_stats(self.__model)

    def validate(self) -> str:
        """Returns a string indicating that the model is invalid."""
        return swig_helper.CpSatHelper.validate_model(self.__model)

    def export_to_file(self, file: str) -> bool:
        """Write the model as a protocol buffer to 'file'.

        Args:
          file: file to write the model to. If the filename ends with 'txt', the
            model will be written as a text file, otherwise, the binary format will
            be used.

        Returns:
          True if the model was correctly written.
        """
        return swig_helper.CpSatHelper.write_model_to_file(self.__model, file)

    def add_hint(self, var: IntVar, value: int) -> None:
        """Adds 'var == value' as a hint to the solver."""
        self.__model.solution_hint.vars.append(self.get_or_make_index(var))
        self.__model.solution_hint.values.append(value)

    def clear_hints(self):
        """Removes any solution hint from the model."""
        self.__model.ClearField('solution_hint')

    def add_assumption(self, lit: LiteralT) -> None:
        """Adds the literal to the model as assumptions."""
        self.__model.assumptions.append(self.get_or_make_boolean_index(lit))

    def add_assumptions(self, literals: Iterable[LiteralT]) -> None:
        """Adds the literals to the model as assumptions."""
        for lit in literals:
            self.add_assumption(lit)

    def clear_assumptions(self) -> None:
        """Removes all assumptions from the model."""
        self.__model.ClearField('assumptions')

    def assert_is_boolean_variable(self, x: LiteralT) -> None:
        if isinstance(x, IntVar):
            var = self.__model.variables[x.index]
            if len(var.domain) != 2 or var.domain[0] < 0 or var.domain[1] > 1:
                raise TypeError('TypeError: ' + str(x) + ' is not a boolean variable')
        elif not isinstance(x, _NotBooleanVariable):
            raise TypeError('TypeError: ' + str(x) + ' is not a boolean variable')

    def Name(self) -> str:
        return self.name

    def SetName(self, name: str) -> None:
        self.name = name

    def Proto(self) -> cp_model_pb2.CpModelProto:
        return self.proto
    NewIntVar = new_int_var
    NewIntVarFromDomain = new_int_var_from_domain
    NewBoolVar = new_bool_var
    NewConstant = new_constant
    NewIntVarSeries = new_int_var_series
    NewBoolVarSeries = new_bool_var_series
    AddLinearConstraint = add_linear_constraint
    AddLinearExpressionInDomain = add_linear_expression_in_domain
    Add = add
    AddAllDifferent = add_all_different
    AddElement = add_element
    AddCircuit = add_circuit
    AddMultipleCircuit = add_multiple_circuit
    AddAllowedAssignments = add_allowed_assignments
    AddForbiddenAssignments = add_forbidden_assignments
    AddAutomaton = add_automaton
    AddInverse = add_inverse
    AddReservoirConstraint = add_reservoir_constraint
    AddImplication = add_implication
    AddBoolOr = add_bool_or
    AddAtLeastOne = add_at_least_one
    AddAtMostOne = add_at_most_one
    AddExactlyOne = add_exactly_one
    AddBoolAnd = add_bool_and
    AddBoolXOr = add_bool_xor
    AddMinEquality = add_min_equality
    AddMaxEquality = add_max_equality
    AddDivisionEquality = add_division_equality
    AddAbsEquality = add_abs_equality
    AddModuloEquality = add_modulo_equality
    AddMultiplicationEquality = add_multiplication_equality
    NewIntervalVar = new_interval_var
    NewIntervalVarSeries = new_interval_var_series
    NewFixedSizedIntervalVar = new_fixed_size_interval_var
    NewOptionalIntervalVar = new_optional_interval_var
    NewOptionalIntervalVarSeries = new_optional_interval_var_series
    NewOptionalFixedSizedIntervalVar = new_optional_fixed_size_interval_var
    NewOptionalFixedSizedIntervalVarSeries = new_optional_fixed_size_interval_var_series
    AddNoOverlap = add_no_overlap
    AddNoOverlap2D = add_no_overlap_2d
    AddCumulative = add_cumulative
    Clone = clone
    GetBoolVarFromProtoIndex = get_bool_var_from_proto_index
    GetIntVarFromProtoIndex = get_int_var_from_proto_index
    GetIntervalVarFromProtoIndex = get_interval_var_from_proto_index
    Minimize = minimize
    Maximize = maximize
    HasObjective = has_objective
    ClearObjective = clear_objective
    AddDecisionStrategy = add_decision_strategy
    ModelStats = model_stats
    Validate = validate
    ExportToFile = export_to_file
    AddHint = add_hint
    ClearHints = clear_hints
    AddAssumption = add_assumption
    AddAssumptions = add_assumptions
    ClearAssumptions = clear_assumptions