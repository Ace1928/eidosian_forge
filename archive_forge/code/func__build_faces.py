import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def _build_faces(self):
    self.faces = []
    edge_sides = set([(e, e.head) for e in self.edges] + [(e, e.tail) for e in self.edges])
    while len(edge_sides):
        es = edge_sides.pop()
        face = OrthogonalFace(self, es)
        edge_sides.difference_update(face)
        self.faces.append(face)