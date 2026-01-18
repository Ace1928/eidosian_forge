import networkx as nx
from .. import t3mlite as t3m
from ..t3mlite.simplex import *
from . import surface
class LinkSurface(surface.Surface):

    def __init__(self, t3m_triangulation):
        self.parent_triangulation = t3m_triangulation
        N = t3m_triangulation
        triangles = []
        for T in N.Tetrahedra:
            new_tris = [surface.Triangle() for i in range(4)]
            triangles += new_tris
            T.CuspCorners = {V0: new_tris[0], V1: new_tris[1], V2: new_tris[2], V3: new_tris[3]}
        surface.Surface.__init__(self, triangles)
        for F in N.Faces:
            F0 = F.Corners[0]
            T0 = F0.Tetrahedron
            f0 = F0.Subsimplex
            v0 = comp(f0)
            F1 = F.Corners[1]
            T1 = F1.Tetrahedron
            f1 = F1.Subsimplex
            v1 = comp(f1)
            for v in VerticesOfFace[f0]:
                w = T0.Gluing[f0].image(v)
                C0, C1 = (T0.CuspCorners[v], T1.CuspCorners[w])
                x0 = cusp_corner_label(v, v0)
                x1 = cusp_corner_label(w, v1)
                E = self.glue_triangles(C0, x0, C1, x1)
                E.face_index = F.Index
                E.reversed = False
        self.build()
        self.label_vertices()

    def label_vertices(self):
        N = self.parent_triangulation
        for vert in self.vertices:
            vert.index = None
        for edge in N.Edges:
            corner = edge.Corners[0]
            tet = corner.Tetrahedron
            a = Head[corner.Subsimplex]
            b = Tail[corner.Subsimplex]
            sign = edge.orientation_with_respect_to(tet, a, b)
            for v, s in [(a, sign), (b, -sign)]:
                label = s * (edge.Index + 1)
                i = TruncatedSimplexCorners[v].index(a | b)
                tri = tet.CuspCorners[v]
                tri.vertices[i].index = label

    def edge_graph(self):
        G = nx.Graph()
        G.add_edges_from([[v.index for v in e.vertices] for e in self.edges])
        return G