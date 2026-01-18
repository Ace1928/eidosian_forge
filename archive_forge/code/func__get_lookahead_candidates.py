import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def _get_lookahead_candidates(self):
    """
        Returns a mapping of {subgraph node: collection of graph nodes} for
        which the graph nodes are feasible candidates for the subgraph node, as
        determined by looking ahead one edge.
        """
    g_counts = {}
    for gn in self.graph:
        g_counts[gn] = self._find_neighbor_color_count(self.graph, gn, self._gn_colors, self._ge_colors)
    candidates = defaultdict(set)
    for sgn in self.subgraph:
        sg_count = self._find_neighbor_color_count(self.subgraph, sgn, self._sgn_colors, self._sge_colors)
        new_sg_count = Counter()
        for (sge_color, sgn_color), count in sg_count.items():
            try:
                ge_color = self._edge_compatibility[sge_color]
                gn_color = self._node_compatibility[sgn_color]
            except KeyError:
                pass
            else:
                new_sg_count[ge_color, gn_color] = count
        for gn, g_count in g_counts.items():
            if all((new_sg_count[x] <= g_count[x] for x in new_sg_count)):
                candidates[sgn].add(gn)
    return candidates