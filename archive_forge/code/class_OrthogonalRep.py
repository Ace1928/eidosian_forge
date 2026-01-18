import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
class OrthogonalRep(Digraph):
    """
    An orthogonal representation is an equivalence class of planar
    embeddings of a graph where all edges are either vertical or
    horizontal. We assume there are no degree 1 vertices.

    Horizontal edges are oriented to the right, and vertical edges
    oriented upwards.

    >>> square = OrthogonalRep([(0, 1), (3, 2)], [(0, 3), (1,2)])
    """

    def __init__(self, horizonal_pairs=[], vertical_pairs=[]):
        Digraph.__init__(self)
        for a, b in horizonal_pairs:
            self.add_edge(a, b, 'horizontal')
        for a, b in vertical_pairs:
            self.add_edge(a, b, 'vertical')
        self._build_faces()
        self._make_turn_regular()

    def add_edge(self, a, b, kind):
        e = Digraph.add_edge(self, a, b)
        e.kind = kind
        return e

    def link(self, vertex):
        """
        The edges in the link oriented counterclockwise.
        """
        link = self.incidence_dict[vertex]
        link.sort(key=lambda e: (e.tail == vertex, e.kind == 'vertical'))
        return link

    def next_edge_at_vertex(self, edge, vertex):
        """
        The next edge at the vertex
        """
        link = self.link(vertex)
        return link[(link.index(edge) + 1) % len(link)]

    def _build_faces(self):
        self.faces = []
        edge_sides = set([(e, e.head) for e in self.edges] + [(e, e.tail) for e in self.edges])
        while len(edge_sides):
            es = edge_sides.pop()
            face = OrthogonalFace(self, es)
            edge_sides.difference_update(face)
            self.faces.append(face)

    def _make_turn_regular(self):
        dummy = set()
        regular = [F for F in self.faces if F.is_turn_regular()]
        irregular = [F for F in self.faces if not F.is_turn_regular()]
        while len(irregular):
            F = irregular.pop()
            i, j = F.kitty_corner()
            v0, v1 = (F[i][1], F[j][1])
            kind = random.choice(('vertical', 'horizontal'))
            if len([e for e in self.incoming(v0) if e.kind == kind]):
                e = self.add_edge(v0, v1, kind)
            else:
                e = self.add_edge(v1, v0, kind)
            dummy.add(e)
            for v in [v0, v1]:
                F = OrthogonalFace(self, (e, v))
                if F.is_turn_regular():
                    regular.append(F)
                else:
                    irregular.append(F)
        self.faces, self.dummy = (regular, dummy)

    def saturation_edges(self, swap_hor_edges):
        return sum([face.saturation_edges(swap_hor_edges) for face in self.faces], [])

    def DAG_from_direction(self, kind):
        H = Digraph(pairs=[e for e in self.edges if e.kind == kind], singles=self.vertices)
        maximal_chains = H.weak_components()
        vertex_to_chain = element_map(maximal_chains)
        D = Digraph(singles=maximal_chains)
        for e in [e for e in self.edges if e.kind != kind]:
            d = D.add_edge(vertex_to_chain[e.tail], vertex_to_chain[e.head])
            d.dummy = e in self.dummy
        for u, v in self.saturation_edges(False):
            d = D.add_edge(vertex_to_chain[u], vertex_to_chain[v])
            d.dummy = True
        for u, v in self.saturation_edges(True):
            if kind == 'vertical':
                u, v = (v, u)
            d = D.add_edge(vertex_to_chain[u], vertex_to_chain[v])
            d.dummy = True
        D.vertex_to_chain = vertex_to_chain
        return D

    def chain_coordinates(self, kind):
        D = self.DAG_from_direction(kind)
        chain_coors = topological_numbering(D)
        return dict(((v, chain_coors[D.vertex_to_chain[v]]) for v in self.vertices))

    def basic_grid_embedding(self, rotate=False):
        """
        Returns the positions of vertices under the grid embedding.
        """
        V = self.chain_coordinates('horizontal')
        H = self.chain_coordinates('vertical')
        return dict(((v, (H[v], V[v])) for v in self.vertices))

    def show(self, unit=10, labels=True):
        from sage.all import circle, text, line, Graphics
        pos = self.basic_grid_embedding()
        for v, (a, b) in pos.items():
            pos[v] = (unit * a, unit * b)
        if not labels:
            verts = [circle(p, 1, fill=True) for p in pos.values()]
        else:
            verts = [text(repr(v), p, fontsize=20, color='black') for v, p in pos.items()]
            verts += [circle(p, 1.5, fill=False) for p in pos.values()]
        edges = [line([pos[e.tail], pos[e.head]]) for e in self.edges if e not in self.dummy]
        G = sum(verts + edges, Graphics())
        G.axes(False)
        return G