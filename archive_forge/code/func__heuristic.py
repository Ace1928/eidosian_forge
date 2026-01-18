from copy import deepcopy
import networkx as nx
from networkx.algorithms.centrality.betweenness import (
from networkx.utils.decorators import not_implemented_for
def _heuristic(k, root, DF_tree, D, nodes, greedy):
    import numpy as np
    node_p = DF_tree.number_of_nodes() + 1
    node_m = DF_tree.number_of_nodes() + 2
    added_node = DF_tree.nodes[root]['CL'][0]
    DF_tree.add_nodes_from([(node_p, deepcopy(DF_tree.nodes[root]))])
    DF_tree.nodes[node_p]['GM'].append(added_node)
    DF_tree.nodes[node_p]['GBC'] += DF_tree.nodes[node_p]['cont'][added_node]
    root_node = DF_tree.nodes[root]
    for x in nodes:
        for y in nodes:
            dxvy = 0
            dxyv = 0
            dvxy = 0
            if not (root_node['sigma'][x][y] == 0 or root_node['sigma'][x][added_node] == 0 or root_node['sigma'][added_node][y] == 0):
                if D[x][added_node] == D[x][y] + D[y][added_node]:
                    dxyv = root_node['sigma'][x][y] * root_node['sigma'][y][added_node] / root_node['sigma'][x][added_node]
                if D[x][y] == D[x][added_node] + D[added_node][y]:
                    dxvy = root_node['sigma'][x][added_node] * root_node['sigma'][added_node][y] / root_node['sigma'][x][y]
                if D[added_node][y] == D[added_node][x] + D[x][y]:
                    dvxy = root_node['sigma'][added_node][x] * root_node['sigma'][x][y] / root_node['sigma'][added_node][y]
            DF_tree.nodes[node_p]['sigma'][x][y] = root_node['sigma'][x][y] * (1 - dxvy)
            DF_tree.nodes[node_p]['betweenness'][x][y] = root_node['betweenness'][x][y] - root_node['betweenness'][x][y] * dxvy
            if y != added_node:
                DF_tree.nodes[node_p]['betweenness'][x][y] -= root_node['betweenness'][x][added_node] * dxyv
            if x != added_node:
                DF_tree.nodes[node_p]['betweenness'][x][y] -= root_node['betweenness'][added_node][y] * dvxy
    DF_tree.nodes[node_p]['CL'] = [node for _, node in sorted(zip(np.diag(DF_tree.nodes[node_p]['betweenness']), nodes), reverse=True) if node not in DF_tree.nodes[node_p]['GM']]
    DF_tree.nodes[node_p]['cont'] = dict(zip(nodes, np.diag(DF_tree.nodes[node_p]['betweenness'])))
    DF_tree.nodes[node_p]['heu'] = 0
    for i in range(k - len(DF_tree.nodes[node_p]['GM'])):
        DF_tree.nodes[node_p]['heu'] += DF_tree.nodes[node_p]['cont'][DF_tree.nodes[node_p]['CL'][i]]
    if not greedy:
        DF_tree.add_nodes_from([(node_m, deepcopy(DF_tree.nodes[root]))])
        DF_tree.nodes[node_m]['CL'].pop(0)
        DF_tree.nodes[node_m]['cont'].pop(added_node)
        DF_tree.nodes[node_m]['heu'] = 0
        for i in range(k - len(DF_tree.nodes[node_m]['GM'])):
            DF_tree.nodes[node_m]['heu'] += DF_tree.nodes[node_m]['cont'][DF_tree.nodes[node_m]['CL'][i]]
    else:
        node_m = None
    return (node_p, node_m, DF_tree)