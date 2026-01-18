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
def _treemaker(scenlist):
    """
    Makes a scenario tree (avoids dependence on daps)

    Parameters
    ----------
    scenlist (list of `int`): experiment (i.e. scenario) numbers

    Returns
    -------
    a `ConcreteModel` that is the scenario tree
    """
    num_scenarios = len(scenlist)
    m = scenario_tree.tree_structure_model.CreateAbstractScenarioTreeModel()
    m = m.create_instance()
    m.Stages.add('Stage1')
    m.Stages.add('Stage2')
    m.Nodes.add('RootNode')
    for i in scenlist:
        m.Nodes.add('LeafNode_Experiment' + str(i))
        m.Scenarios.add('Experiment' + str(i))
    m.NodeStage['RootNode'] = 'Stage1'
    m.ConditionalProbability['RootNode'] = 1.0
    for node in m.Nodes:
        if node != 'RootNode':
            m.NodeStage[node] = 'Stage2'
            m.Children['RootNode'].add(node)
            m.Children[node].clear()
            m.ConditionalProbability[node] = 1.0 / num_scenarios
            m.ScenarioLeafNode[node.replace('LeafNode_', '')] = node
    return m