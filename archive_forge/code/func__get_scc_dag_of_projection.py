from pyomo.common.deprecation import deprecated
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
from pyomo.common.dependencies import networkx as nx
def _get_scc_dag_of_projection(graph, top_nodes, matching):
    """Return the DAG of strongly connected components of a bipartite graph,
    projected with respect to a perfect matching

    This data structure can be used, for instance, to identify the minimal
    subsystem of constraints and variables necessary to solve a given variable
    or constraint.

    """
    nxc = nx.algorithms.components
    dg = _get_projected_digraph(graph, matching, top_nodes).reverse()
    scc_list = list(nxc.strongly_connected_components(dg))
    n_scc = len(scc_list)
    node_scc_map = {n: idx for idx, scc in enumerate(scc_list) for n in scc}
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_scc))
    for n in dg.nodes:
        source_scc = node_scc_map[n]
        for neighbor in dg[n]:
            target_scc = node_scc_map[neighbor]
            if target_scc != source_scc:
                dag.add_edge(source_scc, target_scc)
    return (scc_list, dag)