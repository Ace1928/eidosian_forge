import abc
from cliff.formatters import base
from networkx.drawing.nx_pydot import write_dot
from networkx.readwrite.graphml import GraphMLWriter
from networkx.readwrite import json_graph
import networkx as nx
class DOTFormatter(GraphFormatter):

    def _write_format(self, graph, stdout):
        write_dot(graph, stdout)