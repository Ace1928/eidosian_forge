from snappy.snap import t3mlite as t3m
from truncatedComplex import *
def _compute_shortest_path(self, tet_and_perm):
    result = []
    while True:
        key = self.get_key(tet_and_perm)
        edge = self.tet_and_perm_to_edge[key]
        if edge is None:
            return result[::-1]
        result.append(edge)
        tet_and_perm = edge.tet_and_perm