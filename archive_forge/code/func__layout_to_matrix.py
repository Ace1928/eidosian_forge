import networkx as nx
def _layout_to_matrix(layout):
    """Create the adjacency matrix for the tree specified by the
    given layout (level sequence)."""
    result = [[0] * len(layout) for i in range(len(layout))]
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
            result[i][j] = result[j][i] = 1
        stack.append(i)
    return result