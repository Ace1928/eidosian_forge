import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def _process_ordered_pair_partitions(self, graph, top_partitions, bottom_partitions, edge_colors, orbits=None, cosets=None):
    """
        Processes ordered pair partitions as per the reference paper. Finds and
        returns all permutations and cosets that leave the graph unchanged.
        """
    if orbits is None:
        orbits = [{node} for node in graph.nodes]
    else:
        orbits = orbits
    if cosets is None:
        cosets = {}
    else:
        cosets = cosets.copy()
    assert all((len(t_p) == len(b_p) for t_p, b_p in zip(top_partitions, bottom_partitions)))
    if all((len(top) == 1 for top in top_partitions)):
        permutations = self._find_permutations(top_partitions, bottom_partitions)
        self._update_orbits(orbits, permutations)
        if permutations:
            return ([permutations], cosets)
        else:
            return ([], cosets)
    permutations = []
    unmapped_nodes = {(node, idx) for idx, t_partition in enumerate(top_partitions) for node in t_partition if len(t_partition) > 1}
    node, pair_idx = min(unmapped_nodes)
    b_partition = bottom_partitions[pair_idx]
    for node2 in sorted(b_partition):
        if len(b_partition) == 1:
            continue
        if node != node2 and any((node in orbit and node2 in orbit for orbit in orbits)):
            continue
        partitions = self._couple_nodes(top_partitions, bottom_partitions, pair_idx, node, node2, graph, edge_colors)
        for opp in partitions:
            new_top_partitions, new_bottom_partitions = opp
            new_perms, new_cosets = self._process_ordered_pair_partitions(graph, new_top_partitions, new_bottom_partitions, edge_colors, orbits, cosets)
            permutations += new_perms
            cosets.update(new_cosets)
    mapped = {k for top, bottom in zip(top_partitions, bottom_partitions) for k in top if len(top) == 1 and top == bottom}
    ks = {k for k in graph.nodes if k < node}
    find_coset = ks <= mapped and node not in cosets
    if find_coset:
        for orbit in orbits:
            if node in orbit:
                cosets[node] = orbit.copy()
    return (permutations, cosets)