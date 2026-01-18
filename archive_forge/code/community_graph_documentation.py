from pyomo.common.dependencies import networkx as nx
from pyomo.core import Constraint, Objective, Var, ComponentMap, SortComponents
from pyomo.core.expr import identify_variables
from pyomo.contrib.community_detection.event_log import _event_log

    Creates a networkX graph of nodes and edges based on a Pyomo optimization model

    This function takes in a Pyomo optimization model, then creates a graphical representation of the model with
    specific features of the graph determined by the user (see Parameters below).

    (This function is designed to be called by detect_communities, but can be used solely for the purpose of
    creating model graphs as well.)

    Parameters
    ----------
    model: Block
        a Pyomo model or block to be used for community detection
    type_of_graph: str
        a string that specifies the type of graph that is created from the model
        'constraint' creates a graph based on constraint nodes,
        'variable' creates a graph based on variable nodes,
        'bipartite' creates a graph based on constraint and variable nodes (bipartite graph).
    with_objective: bool, optional
        a Boolean argument that specifies whether or not the objective function is included in the graph; the
        default is True
    weighted_graph: bool, optional
        a Boolean argument that specifies whether a weighted or unweighted graph is to be created from the Pyomo
        model; the default is True (type_of_graph='bipartite' creates an unweighted graph regardless of this parameter)
    use_only_active_components: bool, optional
        a Boolean argument that specifies whether inactive constraints/objectives are included in the networkX graph

    Returns
    -------
    bipartite_model_graph/projected_model_graph: nx.Graph
        a NetworkX graph with nodes and edges based on the given Pyomo optimization model
    number_component_map: dict
        a dictionary that (deterministically) maps a number to a component in the model
    constraint_variable_map: dict
        a dictionary that maps a numbered constraint to a list of (numbered) variables that appear in the constraint
    