import sys
import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def _find_sources(graph):
    """
    Determine a minimal set of nodes such that the entire graph is reachable
    """
    if graph.is_directed():
        sccs = list(nx.strongly_connected_components(graph))
        scc_graph = nx.condensation(graph, sccs)
        supernode_to_nodes = {sn: [] for sn in scc_graph.nodes()}
        mapping = scc_graph.graph['mapping']
        for n in graph.nodes:
            sn = mapping[n]
            supernode_to_nodes[sn].append(n)
        sources = []
        for sn in scc_graph.nodes():
            if scc_graph.in_degree[sn] == 0:
                scc = supernode_to_nodes[sn]
                node = min(scc, key=lambda n: graph.in_degree[n])
                sources.append(node)
    else:
        sources = [min(cc, key=lambda n: graph.degree[n]) for cc in nx.connected_components(graph)]
        sources = sorted(sources, key=lambda n: graph.degree[n])
    return sources