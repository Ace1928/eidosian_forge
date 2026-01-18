import re
import importlib as im
import logging
import types
import json
from itertools import combinations
from pyomo.common.dependencies import (
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID
import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet
def _experiment_instance_creation_callback(scenario_name, node_names=None, cb_data=None):
    """
    This is going to be called by mpi-sppy or the local EF and it will call into
    the user's model's callback.

    Parameters:
    -----------
    scenario_name: `str` Scenario name should end with a number
    node_names: `None` ( Not used here )
    cb_data : dict with ["callback"], ["BootList"],
              ["theta_names"], ["cb_data"], etc.
              "cb_data" is passed through to user's callback function
                        that is the "callback" value.
              "BootList" is None or bootstrap experiment number list.
                       (called cb_data by mpisppy)


    Returns:
    --------
    instance: `ConcreteModel`
        instantiated scenario

    Note:
    ----
    There is flexibility both in how the function is passed and its signature.
    """
    assert cb_data is not None
    outer_cb_data = cb_data
    scen_num_str = re.compile('(\\d+)$').search(scenario_name).group(1)
    scen_num = int(scen_num_str)
    basename = scenario_name[:-len(scen_num_str)]
    CallbackFunction = outer_cb_data['callback']
    if callable(CallbackFunction):
        callback = CallbackFunction
    else:
        cb_name = CallbackFunction
        if 'CallbackModule' not in outer_cb_data:
            raise RuntimeError('Internal Error: need CallbackModule in parmest callback')
        else:
            modname = outer_cb_data['CallbackModule']
        if isinstance(modname, str):
            cb_module = im.import_module(modname, package=None)
        elif isinstance(modname, types.ModuleType):
            cb_module = modname
        else:
            print('Internal Error: bad CallbackModule')
            raise
        try:
            callback = getattr(cb_module, cb_name)
        except:
            print('Error getting function=' + cb_name + ' from module=' + str(modname))
            raise
    if 'BootList' in outer_cb_data:
        bootlist = outer_cb_data['BootList']
        exp_num = bootlist[scen_num]
    else:
        exp_num = scen_num
    scen_name = basename + str(exp_num)
    cb_data = outer_cb_data['cb_data']
    try:
        instance = callback(experiment_number=exp_num, cb_data=cb_data)
    except TypeError:
        raise RuntimeError('Only one callback signature is supported: callback(experiment_number, cb_data) ')
        '\n        try:\n            instance = callback(scenario_tree_model, scen_name, node_names)\n        except TypeError:  # deprecated signature?\n            try:\n                instance = callback(scen_name, node_names)\n            except:\n                print("Failed to create instance using callback; TypeError+")\n                raise\n        except:\n            print("Failed to create instance using callback.")\n            raise\n        '
    if hasattr(instance, '_mpisppy_node_list'):
        raise RuntimeError(f'scenario for experiment {exp_num} has _mpisppy_node_list')
    nonant_list = [instance.find_component(vstr) for vstr in outer_cb_data['theta_names']]
    if use_mpisppy:
        instance._mpisppy_node_list = [scenario_tree.ScenarioNode(name='ROOT', cond_prob=1.0, stage=1, cost_expression=instance.FirstStageCost, nonant_list=nonant_list, scen_model=instance)]
    else:
        instance._mpisppy_node_list = [scenario_tree.ScenarioNode(name='ROOT', cond_prob=1.0, stage=1, cost_expression=instance.FirstStageCost, scen_name_list=None, nonant_list=nonant_list, scen_model=instance)]
    if 'ThetaVals' in outer_cb_data:
        thetavals = outer_cb_data['ThetaVals']
        for vstr in thetavals:
            theta_cuid = ComponentUID(vstr)
            theta_object = theta_cuid.find_component_on(instance)
            if thetavals[vstr] is not None:
                theta_object.fix(thetavals[vstr])
            else:
                theta_object.unfix()
    return instance