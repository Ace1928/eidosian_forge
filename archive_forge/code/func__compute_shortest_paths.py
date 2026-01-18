from snappy.snap import t3mlite as t3m
from truncatedComplex import *
def _compute_shortest_paths(self):
    base_tet_and_perm = (0, t3m.Perm4([0, 1, 2, 3]))
    self.tet_and_perm_to_edge = {self.get_key(base_tet_and_perm): None}
    pending = [base_tet_and_perm]
    self.alpha_edges = [None for edge in self.mcomplex.Edges]
    while pending:
        tet_and_perm = pending.pop()
        edge_index, edge_end = self.get_edge_index_and_end_from_tet_and_perm(tet_and_perm)
        if edge_end == 0 and self.alpha_edges[edge_index] is None:
            self.alpha_edges[edge_index] = TruncatedComplex.Edge('alpha', tet_and_perm)
        for edge in self.get_edges_for_tet_and_perm(tet_and_perm):
            if edge.subcomplex_type != 'alpha':
                tet_and_perm_of_end = edge.tet_and_perm_of_end()
                key = self.get_key(tet_and_perm_of_end)
                if key not in self.tet_and_perm_to_edge:
                    self.tet_and_perm_to_edge[key] = edge
                    pending.append(tet_and_perm_of_end)