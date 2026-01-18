import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _prim(G):
    """Prim's algorithm to find minimum spanning tree (PRIVATE).

    Code is adapted from
    http://programmingpraxis.com/2010/04/09/minimum-spanning-tree-prims-algorithm/
    """
    nodes = []
    edges = []
    for i in G.keys():
        nodes.append(i)
        for j in G[i]:
            if (i, j, G[i][j]) not in edges and (j, i, G[i][j]) not in edges:
                edges.append((i, j, G[i][j]))
    conn = defaultdict(list)
    for n1, n2, c in edges:
        conn[n1].append((c, n1, n2))
        conn[n2].append((c, n2, n1))
    mst = []
    used = set(nodes[0])
    usable_edges = conn[nodes[0]][:]
    heapify(usable_edges)
    while usable_edges:
        cost, n1, n2 = heappop(usable_edges)
        if n2 not in used:
            used.add(n2)
            mst.append((n1, n2, cost))
            for e in conn[n2]:
                if e[2] not in used:
                    heappush(usable_edges, e)
    length = 0
    for p in mst:
        length += floor(p[2])
    return length