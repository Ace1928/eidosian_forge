import networkx as nx
def _layout_to_graph(layout):
    """Create a NetworkX Graph for the tree specified by the
    given layout(level sequence)"""
    G = nx.Graph()
    stack = []
    for i in range(len(layout)):
        i_level = layout[i]
        if stack:
            j = stack[-1]
            j_level = layout[j]
            while j_level >= i_level:
                stack.pop()
                j = stack[-1]
                j_level = layout[j]
            G.add_edge(i, j)
        stack.append(i)
    return G