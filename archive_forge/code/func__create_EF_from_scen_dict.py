from pyomo.core.expr.numeric_expr import LinearExpression
import pyomo.environ as pyo
from pyomo.core import Objective
def _create_EF_from_scen_dict(scen_dict, EF_name=None, nonant_for_fixed_vars=True):
    """Create a ConcreteModel of the extensive form from a scenario
    dictionary.

    Args:
        scen_dict (dict): Dictionary whose keys are scenario names and
            values are ConcreteModel objects corresponding to each
            scenario.
        EF_name (str--optional): Name of the resulting EF model.
        nonant_for_fixed_vars (bool--optional): If True, enforces
            non-anticipativity constraints for all variables, including
            those which have been fixed. Default is True.

    Returns:
        EF_instance (ConcreteModel): ConcreteModel of extensive form with
            explicitly non-anticipativity constraints.

    Notes:
        The non-anticipativity constraints are enforced by creating
        "reference variables" at each node in the scenario tree (excluding
        leaves) and enforcing that all the variables for each scenario at
        that node are equal to the reference variables.

        This function is called directly when creating bundles for PH.

        Does NOT assume that each scenario is equally likely. Raises an
        AttributeError if a scenario object is encountered which does not
        have a ._mpisppy_probability attribute.

        Added the flag nonant_for_fixed_vars because original code only
        enforced non-anticipativity for non-fixed vars, which is not always
        desirable in the context of bundling. This allows for more
        fine-grained control.
    """
    is_min, clear = _models_have_same_sense(scen_dict)
    if not clear:
        raise RuntimeError('Cannot build the extensive form out of models with different objective senses')
    sense = pyo.minimize if is_min else pyo.maximize
    EF_instance = pyo.ConcreteModel(name=EF_name)
    EF_instance.EF_Obj = pyo.Objective(expr=0.0, sense=sense)
    EF_instance._mpisppy_data = pyo.Block(name='For non-Pyomo mpi-sppy data')
    EF_instance._mpisppy_model = pyo.Block(name='For mpi-sppy Pyomo additions to the scenario model')
    EF_instance._mpisppy_data.scenario_feasible = None
    EF_instance._ef_scenario_names = []
    EF_instance._mpisppy_probability = 0
    for sname, scenario_instance in scen_dict.items():
        EF_instance.add_component(sname, scenario_instance)
        EF_instance._ef_scenario_names.append(sname)
        scenario_objs = get_objs(scenario_instance)
        for obj_func in scenario_objs:
            obj_func.deactivate()
        obj_func = scenario_objs[0]
        try:
            EF_instance.EF_Obj.expr += scenario_instance._mpisppy_probability * obj_func.expr
            EF_instance._mpisppy_probability += scenario_instance._mpisppy_probability
        except AttributeError as e:
            raise AttributeError('Scenario ' + sname + ' has no specified probability. Specify a value for the attribute  _mpisppy_probability and try again.') from e
    EF_instance.EF_Obj.expr /= EF_instance._mpisppy_probability
    ref_vars = dict()
    ref_suppl_vars = dict()
    EF_instance._nlens = dict()
    nonant_constr = pyo.Constraint(pyo.Any, name='_C_EF_')
    EF_instance.add_component('_C_EF_', nonant_constr)
    nonant_constr_suppl = pyo.Constraint(pyo.Any, name='_C_EF_suppl')
    EF_instance.add_component('_C_EF_suppl', nonant_constr_suppl)
    for sname, s in scen_dict.items():
        nlens = {node.name: len(node.nonant_vardata_list) for node in s._mpisppy_node_list}
        for node_name, num_nonant_vars in nlens.items():
            if node_name in EF_instance._nlens.keys() and num_nonant_vars != EF_instance._nlens[node_name]:
                raise RuntimeError('Number of non-anticipative variables is not consistent at node ' + node_name + ' in scenario ' + sname)
            EF_instance._nlens[node_name] = num_nonant_vars
        nlens_ef_suppl = {node.name: len(node.nonant_ef_suppl_vardata_list) for node in s._mpisppy_node_list}
        for node in s._mpisppy_node_list:
            ndn = node.name
            for i in range(nlens[ndn]):
                v = node.nonant_vardata_list[i]
                if (ndn, i) not in ref_vars:
                    ref_vars[ndn, i] = v
                elif nonant_for_fixed_vars or not v.is_fixed():
                    expr = LinearExpression(linear_coefs=[1, -1], linear_vars=[v, ref_vars[ndn, i]], constant=0.0)
                    nonant_constr[ndn, i, sname] = (expr, 0.0)
            for i in range(nlens_ef_suppl[ndn]):
                v = node.nonant_ef_suppl_vardata_list[i]
                if (ndn, i) not in ref_suppl_vars:
                    ref_suppl_vars[ndn, i] = v
                elif nonant_for_fixed_vars or not v.is_fixed():
                    expr = LinearExpression(linear_coefs=[1, -1], linear_vars=[v, ref_suppl_vars[ndn, i]], constant=0.0)
                    nonant_constr_suppl[ndn, i, sname] = (expr, 0.0)
    EF_instance.ref_vars = ref_vars
    EF_instance.ref_suppl_vars = ref_suppl_vars
    return EF_instance