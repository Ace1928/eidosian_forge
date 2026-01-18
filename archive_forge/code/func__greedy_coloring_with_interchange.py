import itertools
from collections import defaultdict, deque
import networkx as nx
from networkx.utils import arbitrary_element, py_random_state
def _greedy_coloring_with_interchange(G, nodes):
    """Return a coloring for `original_graph` using interchange approach

    This procedure is an adaption of the algorithm described by [1]_,
    and is an implementation of coloring with interchange. Please be
    advised, that the datastructures used are rather complex because
    they are optimized to minimize the time spent identifying
    subcomponents of the graph, which are possible candidates for color
    interchange.

    Parameters
    ----------
    G : NetworkX graph
        The graph to be colored

    nodes : list
        nodes ordered using the strategy of choice

    Returns
    -------
    dict :
        A dictionary keyed by node to a color value

    References
    ----------
    .. [1] Maciej M. Syslo, Narsingh Deo, Janusz S. Kowalik,
       Discrete Optimization Algorithms with Pascal Programs, 415-424, 1983.
       ISBN 0-486-45353-7.
    """
    n = len(G)
    graph = {node: _Node(node, n) for node in G}
    for node1, node2 in G.edges():
        adj_entry1 = _AdjEntry(node2)
        adj_entry2 = _AdjEntry(node1)
        adj_entry1.mate = adj_entry2
        adj_entry2.mate = adj_entry1
        node1_head = graph[node1].adj_list
        adj_entry1.next = node1_head
        graph[node1].adj_list = adj_entry1
        node2_head = graph[node2].adj_list
        adj_entry2.next = node2_head
        graph[node2].adj_list = adj_entry2
    k = 0
    for node in nodes:
        neighbors = graph[node].iter_neighbors()
        col_used = {graph[adj_node.node_id].color for adj_node in neighbors}
        col_used.discard(-1)
        k1 = next(itertools.dropwhile(lambda x: x in col_used, itertools.count()))
        if k1 > k:
            connected = True
            visited = set()
            col1 = -1
            col2 = -1
            while connected and col1 < k:
                col1 += 1
                neighbor_cols = graph[node].iter_neighbors_color(col1)
                col1_adj = list(neighbor_cols)
                col2 = col1
                while connected and col2 < k:
                    col2 += 1
                    visited = set(col1_adj)
                    frontier = list(col1_adj)
                    i = 0
                    while i < len(frontier):
                        search_node = frontier[i]
                        i += 1
                        col_opp = col2 if graph[search_node].color == col1 else col1
                        neighbor_cols = graph[search_node].iter_neighbors_color(col_opp)
                        for neighbor in neighbor_cols:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                frontier.append(neighbor)
                    connected = len(visited.intersection(graph[node].iter_neighbors_color(col2))) > 0
            if not connected:
                for search_node in visited:
                    graph[search_node].color = col2 if graph[search_node].color == col1 else col1
                    col2_adj = graph[search_node].adj_color[col2]
                    graph[search_node].adj_color[col2] = graph[search_node].adj_color[col1]
                    graph[search_node].adj_color[col1] = col2_adj
                for search_node in visited:
                    col = graph[search_node].color
                    col_opp = col1 if col == col2 else col2
                    for adj_node in graph[search_node].iter_neighbors():
                        if graph[adj_node.node_id].color != col_opp:
                            adj_mate = adj_node.mate
                            graph[adj_node.node_id].clear_color(adj_mate, col_opp)
                            graph[adj_node.node_id].assign_color(adj_mate, col)
                k1 = col1
        graph[node].color = k1
        k = max(k1, k)
        for adj_node in graph[node].iter_neighbors():
            adj_mate = adj_node.mate
            graph[adj_node.node_id].assign_color(adj_mate, k1)
    return {node.node_id: node.color for node in graph.values()}