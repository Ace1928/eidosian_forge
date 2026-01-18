import networkx as nx
from networkx import DiGraph, Graph, MultiDiGraph, MultiGraph, PlanarEmbedding
from networkx.classes.reportviews import NodeView
class LoopbackDiGraph(DiGraph):
    __networkx_backend__ = 'nx-loopback'