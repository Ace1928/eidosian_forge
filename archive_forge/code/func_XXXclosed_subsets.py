import networkx as nx
from collections import deque
def XXXclosed_subsets(self, start=None):
    """
        Generator for all transitively closed subsets.  The subsets
        are computed once, then cached for use in subsequent calls.

        >>> G = Digraph([(0,1),(1,2),(2,4),(0,3),(3,4)])
        >>> P = Poset(G)
        >>> len(list(P.XXXclosed_subsets()))
        7
        """
    if start is None:
        if self.closed:
            yield from self.closed
            return
        else:
            start = self.smallest()
    if start not in self.closed:
        self.closed.add(start)
        yield start
    children = set()
    for element in start:
        children.update(self.successors[element] - start)
    for child in children:
        extended = self.closure(start | set([child]))
        yield from self.XXXclosed_subsets(extended)