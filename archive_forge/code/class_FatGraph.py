import networkx as nx
from collections import deque
class FatGraph(Graph):
    """
    A FatGraph is a Graph which maintains a CyclicList of incident
    edges for each vertex.  The edges are FatEdges, which come in two
    flavors: twisted and untwisted.  Since the incident edges are in
    particular cyclically ordered, a FatGraph has a canonical
    embedding as a spine of a surface with boundary.  However, a
    CyclicList has a first element.  This extra data, namely a choice
    of distinguished edge for each vertex, is used to give a canonical
    mapping from FatGraphs to link or tangle diagrams.

    A FatGraph should be initialized with a sequence of either double
    pairs ((v,n), (w,m)) or triples ((v,n), (w,m), T).  These indicate
    that there is an edge from v to w which has index n at v and m at
    w.  If T is supplied then the parity of T determines the twist.
    """
    edge_class = FatEdge

    def __call__(self, vertex):
        return self.incidence_dict[vertex]

    def add_edge(self, *args):
        incidences = self.incidence_dict
        edge = self.Edge(*args)
        self.edges.add(edge)
        self.vertices.update(edge)
        for v in edge:
            if v in incidences:
                incidences[v].append(edge)
                incidences[v].sort(key=lambda e: e.slot(v))
            else:
                incidences[v] = CyclicList([edge])
        return edge

    def _validate(self):
        for v in self.vertices:
            slots = [e.slot(v) for e in self(v)]
            assert slots == range(len(slots))

    def reorder(self, vertex, cyclist):
        for e, n in zip(self(vertex), cyclist):
            e.set_slot(vertex, n)
        self.incidence_dict[vertex].sort(key=lambda e: e.slot(vertex))

    def boundary_cycles(self):
        left = [(e[0], e, 'L') for e in self.edges]
        right = [(e[0], e, 'R') for e in self.edges]
        sides = left + right
        cycles = []
        while sides:
            cycle = []
            v, e, s = start = sides.pop()
            while True:
                cycle.append(e)
                v = e(v)
                if e.twisted and s == 'L' or (not e.twisted and (s == 'L') == (v == e[0])):
                    e = self(v).succ(e)
                    s = 'R' if e.twisted or v == e[0] else 'L'
                else:
                    e = self(v).pred(e)
                    s = 'L' if e.twisted or v == e[0] else 'R'
                if (e[0], e, s) == start:
                    cycles.append(cycle)
                    break
                sides.remove((e[0], e, s))
        return cycles

    def filled_euler(self):
        return len(self.vertices) - len(self.edges) + len(self.boundary_cycles())