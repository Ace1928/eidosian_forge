import networkx as nx
from collections import deque
def XXclosed_subsets(self, start=None):
    """
        Generator for all transitively closed subsets.

        >>> G = Digraph([(0,1),(1,2),(2,4),(0,3),(3,4)])
        >>> P = Poset(G)
        >>> len(list(P.XXclosed_subsets()))
        7
        """
    if start is None:
        if self.closed:
            yield from self.closed
            return
        else:
            start = self.smallest()
    complement = self.elements - start
    if start not in self.closed:
        self.closed.add(start)
        yield start
    for element in complement:
        extended = self.closure(start | set([element]))
        yield from self.XXclosed_subsets(extended)