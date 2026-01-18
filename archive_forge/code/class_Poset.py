import networkx as nx
from collections import deque
class Poset(set):
    """
    A partially ordered set, generated from a directed acyclic graph.

    Instantiate with a Digraph. A :class:`ValueError` exception is raised
    if the Digraph contains a cycle.
    """

    def __init__(self, digraph):
        self.elements = set(digraph.vertices)
        self.larger = {vertex: set() for vertex in self}
        self.smaller = {vertex: set() for vertex in self}
        self.successors = {vertex: set(digraph[vertex]) for vertex in self}
        self.closed = set()
        seen = set()
        for vertex in self:
            if vertex not in seen:
                self.search(vertex, seen, digraph)

    def __iter__(self):
        return self.elements.__iter__()

    def __len__(self):
        return len(self.elements)

    def search(self, vertex, seen, digraph):
        seen.add(vertex)
        for child in digraph.children(vertex):
            if child in self.smaller[vertex]:
                raise ValueError('Digraph is not acyclic.')
            self.smaller[child].add(vertex)
            self.smaller[child] |= self.smaller[vertex]
            self.search(child, seen, digraph)
            self.larger[vertex].add(child)
            self.larger[vertex] |= self.larger[child]

    def compare(self, x, y):
        if x == y:
            return 0
        if x in self.smaller[y]:
            return 1
        if y in self.smaller[x]:
            return -1
        return None

    def incomparable(self, x):
        """
        Return the elements which are not comparable to x.

        >>> G = Digraph([(0,1),(1,2),(2,4),(0,3),(3,4)])
        >>> P = Poset(G)
        >>> sorted(P.incomparable(3))
        [1, 2]
        """
        return self.elements - self.smaller[x] - self.larger[x] - set([x])

    def smallest(self):
        """
        Return the subset of minimal elements.

        >>> G = Digraph([(0,1),(1,2),(2,4),(0,3),(3,4)])
        >>> P = Poset(G)
        >>> sorted(P.smallest())
        [0]
        """
        return frozenset([x for x in self if not self.smaller[x]])

    def largest(self):
        """
        Return the subset of maximal elements.

        >>> G = Digraph([(0,1),(1,2),(2,4),(0,3),(3,4)])
        >>> P = Poset(G)
        >>> sorted(P.largest())
        [4]
        """
        return frozenset([x for x in self if not self.larger[x]])

    def closure(self, A):
        """
        Return the smallest set X containing A which is is closed
        under < , i.e. such that (x in X and y < x) => y in X.

        >>> G = Digraph([(0,1),(1,2),(2,4),(0,3),(3,4)])
        >>> P = Poset(G)
        >>> sorted(P.closure([1, 3]))
        [0, 1, 3]
        """
        result = frozenset(A)
        for a in A:
            result |= self.smaller[a]
        return result

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

    def closed_subsets(self):
        """
        Generator for all nonempty transitively closed subsets.

        >>> G = Digraph([(0,1),(1,2),(2,4),(0,3),(3,4)])
        >>> P = Poset(G)
        >>> len(list(P.closed_subsets()))
        7
        """
        for X in powerset(self.elements):
            if not X:
                continue
            pairwise_incomparable = True
            for x in X:
                if any((y in self.smaller[x] or y in self.larger[x] for y in X)):
                    pairwise_incomparable = False
                    break
            if pairwise_incomparable:
                yield self.closure(X)