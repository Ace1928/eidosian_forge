import networkx as nx
from collections import deque
class ReducedGraph(Graph):
    """
    A graph with at most one edge between any two vertices,
    but having edges with multiplicities.
    """
    edge_class = MultiEdge

    def __init__(self, pairs=[], singles=[]):
        self.Edge = self.__class__.edge_class
        self.find_edge = {}
        Graph.__init__(self, pairs, singles)

    def add_edge(self, x, y):
        if (x, y) in self.find_edge:
            edge = self.find_edge[x, y]
            edge.multiplicity += 1
        else:
            edge = self.Edge(x, y)
            self.vertices.update([x, y])
            self.edges.add(edge)
            self.find_edge[x, y] = edge
            for v in edge:
                try:
                    self.incidence_dict[v].append(edge)
                except KeyError:
                    self.incidence_dict[v] = [edge]
        return edge

    def multi_valence(self, vertex):
        """
        Return the valence of a vertex, counting edge multiplicities.
        """
        return sum((e.multiplicity for e in self.incidence_dict[vertex]))

    def is_planar(self):
        """
        Return the planarity.
        """
        verts_with_loops = set()
        non_loop_edges = set()
        for e in self.edges:
            v, w = e
            if v != w:
                non_loop_edges.add((v, w))
            else:
                verts_with_loops.add(v)
        sans_loops = ReducedGraph(non_loop_edges)
        if _within_sage:
            S = sans_loops.sage(loops=False, multiedges=False)
            is_planar = S.is_planar(set_embedding=True)
            embedding = S.get_embedding() if is_planar else None
        else:
            is_planar, embedding = planar(sans_loops)
        if is_planar:
            for v in verts_with_loops:
                embedding[v].append(v)
            self._embedding = embedding
        return is_planar

    def embedding(self):
        if self.is_planar():
            return self._embedding

    def one_min_cut(self, source, sink):
        capacity = {e: e.multiplicity for e in self.edges}
        cut = Graph.one_min_cut(self, source, sink, capacity)
        cut['size'] = sum((e.multiplicity for e in cut['edges']))
        return cut

    def cut_pairs(self):
        """
        Return a list of cut_pairs.  The graph is assumed to be
        connected and to have no cut vertices.
        """
        pairs = []
        majors = [v for v in self.vertices if self.valence(v) > 2]
        if len(majors) == 2:
            v, V = majors
            if self.valence(v) == 3:
                return []
            edge = self.find_edge[v, V]
            if not edge or edge.multiplicity < 2:
                return []
            return majors
        major_set = set(majors)
        for n in range(1, len(majors)):
            for v in majors[:n]:
                pair = (v, majors[n])
                components = self.components(deleted_vertices=pair)
                if len(components) > 2:
                    pairs.append(pair)
                elif len(components) == 2:
                    M0 = len(major_set & components[0])
                    M1 = len(major_set & components[1])
                    edge = self.find_edge[pair]
                    if edge:
                        if M0 or M1:
                            pairs.append(pair)
                            continue
                    elif M0 and M1:
                        pairs.append(pair)
        return pairs