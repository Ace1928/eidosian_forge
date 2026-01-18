from . import isomorphvf2 as vf2
def _semantic_feasibility(self, G1_node, G2_node):
    """Returns True if mapping G1_node to G2_node is semantically feasible."""
    if self.node_match is not None:
        nm = self.node_match(self.G1.nodes[G1_node], self.G2.nodes[G2_node])
        if not nm:
            return False
    if self.edge_match is not None:
        G1nbrs = self.G1_adj[G1_node]
        G2nbrs = self.G2_adj[G2_node]
        core_1 = self.core_1
        edge_match = self.edge_match
        for neighbor in G1nbrs:
            if neighbor == G1_node:
                if G2_node in G2nbrs and (not edge_match(G1nbrs[G1_node], G2nbrs[G2_node])):
                    return False
            elif neighbor in core_1:
                G2_nbr = core_1[neighbor]
                if G2_nbr in G2nbrs and (not edge_match(G1nbrs[neighbor], G2nbrs[G2_nbr])):
                    return False
    return True