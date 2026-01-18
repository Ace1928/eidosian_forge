import pytest
import networkx as nx
class TestDirectedTreeRecognition(TestTreeRecognition):
    graph = nx.DiGraph
    multigraph = nx.MultiDiGraph