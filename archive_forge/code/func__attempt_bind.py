import collections
import copy
import itertools
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagdependency import DAGDependency
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag
from qiskit.utils import optionals as _optionals
def _attempt_bind(self, template_sublist, circuit_sublist):
    """
        Copies the template and attempts to bind any parameters,
        i.e. attempts to solve for a valid parameter assignment.
        template_sublist and circuit_sublist match up to the
        assignment of the parameters. For example the template

        .. parsed-literal::

                 ┌───────────┐                  ┌────────┐
            q_0: ┤ P(-1.0*β) ├──■────────────■──┤0       ├
                 ├───────────┤┌─┴─┐┌──────┐┌─┴─┐│  CZ(β) │
            q_1: ┤ P(-1.0*β) ├┤ X ├┤ P(β) ├┤ X ├┤1       ├
                 └───────────┘└───┘└──────┘└───┘└────────┘

        should only maximally match once in the circuit

        .. parsed-literal::

                 ┌───────┐
            q_0: ┤ P(-2) ├──■────────────■────────────────────────────
                 ├───────┤┌─┴─┐┌──────┐┌─┴─┐┌──────┐
            q_1: ┤ P(-2) ├┤ X ├┤ P(2) ├┤ X ├┤ P(3) ├──■────────────■──
                 └┬──────┤└───┘└──────┘└───┘└──────┘┌─┴─┐┌──────┐┌─┴─┐
            q_2: ─┤ P(3) ├──────────────────────────┤ X ├┤ P(3) ├┤ X ├
                  └──────┘                          └───┘└──────┘└───┘

        However, up until attempt bind is called, the soft matching
        will have found two matches due to the parameters.
        The first match can be satisfied with β=2. However, the
        second match would imply both β=3 and β=-3 which is impossible.
        Attempt bind detects inconsistencies by solving a system of equations
        given by the parameter expressions in the sub-template and the
        value of the parameters in the gates of the sub-circuit. If a
        solution is found then the match is valid and the parameters
        are assigned. If not, None is returned.

        In order to resolve the conflict of the same parameter names in the
        circuit and template, each variable in the template sublist is
        re-assigned to a new dummy parameter with a completely separate name
        if it clashes with one that exists in an input circuit.

        Args:
            template_sublist (list): part of the matched template.
            circuit_sublist (list): part of the matched circuit.

        Returns:
            DAGDependency: A deep copy of the template with
                the parameters bound. If no binding satisfies the
                parameter constraints, returns None.
        """
    import sympy as sym
    from sympy.parsing.sympy_parser import parse_expr
    if _optionals.HAS_SYMENGINE:
        import symengine
        to_native_symbolic = symengine.sympify
    else:
        to_native_symbolic = lambda x: x
    circuit_params, template_params = ([], [])
    circuit_params_set = set()
    template_dag_dep = copy.deepcopy(self.template_dag_dep)
    for idx, _ in enumerate(template_sublist):
        qc_idx = circuit_sublist[idx]
        parameters = self.circuit_dag_dep.get_node(qc_idx).op.params
        circuit_params += parameters
        for parameter in parameters:
            if isinstance(parameter, ParameterExpression):
                circuit_params_set.update((x.name for x in parameter.parameters))
    _dummy_counter = itertools.count()

    def dummy_parameter():
        return Parameter(f'_qiskit_template_dummy_{next(_dummy_counter)}')
    template_clash_substitutions = collections.defaultdict(dummy_parameter)
    for t_idx in template_sublist:
        node = template_dag_dep.get_node(t_idx)
        sub_node_params = []
        for t_param_exp in node.op.params:
            if isinstance(t_param_exp, ParameterExpression):
                for t_param in t_param_exp.parameters:
                    if t_param.name in circuit_params_set:
                        new_param = template_clash_substitutions[t_param.name]
                        t_param_exp = t_param_exp.assign(t_param, new_param)
            sub_node_params.append(t_param_exp)
            template_params.append(t_param_exp)
        if not node.op.mutable:
            node.op = node.op.to_mutable()
        node.op.params = sub_node_params
    for node in template_dag_dep.get_nodes():
        sub_node_params = []
        for param_exp in node.op.params:
            if isinstance(param_exp, ParameterExpression):
                for param in param_exp.parameters:
                    if param.name in template_clash_substitutions:
                        param_exp = param_exp.assign(param, template_clash_substitutions[param.name])
            sub_node_params.append(param_exp)
        if not node.op.mutable:
            node.op = node.op.to_mutable()
        node.op.params = sub_node_params
    equations, circ_dict, temp_symbols = ([], {}, {})
    for circuit_param, template_param in zip(circuit_params, template_params):
        if isinstance(template_param, ParameterExpression):
            if isinstance(circuit_param, ParameterExpression):
                circ_param_sym = circuit_param.sympify()
            else:
                circ_param_sym = parse_expr(str(circuit_param))
            equations.append(sym.Eq(template_param.sympify(), circ_param_sym))
            for param in template_param.parameters:
                temp_symbols[param] = param.sympify()
            if isinstance(circuit_param, ParameterExpression):
                for param in circuit_param.parameters:
                    circ_dict[param] = param.sympify()
        elif template_param != circuit_param:
            return None
    if not temp_symbols:
        return template_dag_dep
    sym_sol = sym.solve(equations, set(temp_symbols.values()), dict=True)
    if not sym_sol:
        return None
    sol = {param.name: ParameterExpression(circ_dict, to_native_symbolic(expr)) for param, expr in sym_sol[0].items()}
    fake_bind = {key: sol[key.name] for key in temp_symbols}
    for node in template_dag_dep.get_nodes():
        bound_params = []
        for param_exp in node.op.params:
            if isinstance(param_exp, ParameterExpression):
                for param in param_exp.parameters:
                    if param in fake_bind:
                        if fake_bind[param] not in bound_params:
                            param_exp = param_exp.assign(param, fake_bind[param])
            else:
                param_exp = float(param_exp)
            bound_params.append(param_exp)
        if not node.op.mutable:
            node.op = node.op.to_mutable()
        node.op.params = bound_params
    return template_dag_dep