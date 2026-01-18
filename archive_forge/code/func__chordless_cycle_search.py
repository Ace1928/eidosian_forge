from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def _chordless_cycle_search(F, B, path, length_bound):
    """The main loop for chordless cycle enumeration.

    This algorithm is strongly inspired by that of Dias et al [1]_.  It has been
    modified in the following ways:

        1. Recursion is avoided, per Python's limitations

        2. The labeling function is not necessary, because the starting paths
            are chosen (and deleted from the host graph) to prevent multiple
            occurrences of the same path

        3. The search is optionally bounded at a specified length

        4. Support for directed graphs is provided by extending cycles along
            forward edges, and blocking nodes along forward and reverse edges

        5. Support for multigraphs is provided by omitting digons from the set
            of forward edges

    Parameters
    ----------
    F : _NeighborhoodCache
       A graph of forward edges to follow in constructing cycles

    B : _NeighborhoodCache
       A graph of blocking edges to prevent the production of chordless cycles

    path : list
       A cycle prefix.  All cycles generated will begin with this prefix.

    length_bound : int
       A length bound.  All cycles generated will have length at most length_bound.


    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    References
    ----------
    .. [1] Efficient enumeration of chordless cycles
       E. Dias and D. Castonguay and H. Longo and W.A.R. Jradi
       https://arxiv.org/abs/1309.1051

    """
    blocked = defaultdict(int)
    target = path[0]
    blocked[path[1]] = 1
    for w in path[1:]:
        for v in B[w]:
            blocked[v] += 1
    stack = [iter(F[path[2]])]
    while stack:
        nbrs = stack[-1]
        for w in nbrs:
            if blocked[w] == 1 and (length_bound is None or len(path) < length_bound):
                Fw = F[w]
                if target in Fw:
                    yield (path + [w])
                else:
                    Bw = B[w]
                    if target in Bw:
                        continue
                    for v in Bw:
                        blocked[v] += 1
                    path.append(w)
                    stack.append(iter(Fw))
                    break
        else:
            stack.pop()
            for v in B[path.pop()]:
                blocked[v] -= 1