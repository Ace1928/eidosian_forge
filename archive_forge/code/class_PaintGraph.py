from . import schema
from .jsonutil import get_column
from .search import Search
class PaintGraph(object):

    def __init__(self, interface):
        self._intf = interface
        self.get_graph = interface._get_graph

    def architecture(self, with_datatypes=True, save=None):
        graph = self.get_graph.architecture(with_datatypes)
        plt.figure(figsize=(8, 8))
        pos = graphviz_layout(graph, prog='twopi', args='')
        cost = lambda v: float(graph.degree(v)) ** 3 + graph.weights[v] ** 2
        costs = norm_costs([cost(v) for v in graph], 10000)
        nx.draw(graph, pos, labels=graph.labels, node_size=costs, node_color=costs, font_size=13, font_color='orange', font_weight='bold', with_labels=True)
        plt.axis('off')
        if save is not None:
            plt.savefig(save)
        plt.show()

    def experiments(self, save=None):
        graph = self.get_graph.rest_resource('experiments')
        self._draw_rest_resource(graph, save=None)

    def assessors(self, save=None):
        graph = self.get_graph.rest_resource('assessors')
        self._draw_rest_resource(graph, save=None)

    def reconstructions(self, save=None):
        graph = self.get_graph.rest_resource('reconstructions')
        self._draw_rest_resource(graph, save=None)

    def scans(self):
        graph = self.get_graph.rest_resource('scans')
        self._draw_rest_resource(graph)

    def _draw_rest_resource(self, graph, save=None):
        plt.figure(figsize=(8, 8))
        pos = graphviz_layout(graph, prog='twopi', args='')
        cost = lambda v: float(graph.degree(v)) ** 3 + graph.weights[v] ** 2
        node_size = [cost(v) for v in graph]
        node_color = [cost(v) for v in graph]
        nx.draw(graph, pos, node_size=node_size, node_color=node_color, font_size=13, font_color='green', font_weight='bold')
        plt.axis('off')
        if save is not None:
            plt.savefig(save)
        plt.show()

    def datatypes(self, pattern='*', save=None):
        graph = self.get_graph.datatypes(pattern)
        plt.figure(figsize=(8, 8))
        pos = graphviz_layout(graph, prog='twopi', args='')
        cost = lambda v: float(graph.degree(v)) ** 3 + graph.weights[v] ** 2
        node_size = [cost(v) for v in graph]
        node_color = [cost(v) for v in graph]
        nx.draw(graph, pos, node_size=node_size, node_color=node_color, font_size=13, font_color='green', font_weight='bold', with_labels=True)
        plt.axis('off')
        if save is not None:
            plt.savefig(save)
        plt.show()

    def field_values(self, field_name, save=None):
        graph = self.get_graph.field_values(field_name)
        plt.figure(figsize=(8, 8))
        pos = graphviz_layout(graph, prog='twopi', args='')
        cost = lambda v: graph.weights[v]
        graph.weights[field_name] = max([cost(v) for v in graph]) / 2.0
        costs = norm_costs([cost(v) for v in graph], 10000)
        nx.draw(graph, pos, node_size=costs, node_color=costs, font_size=13, font_color='black', font_weight='bold', with_labels=True)
        plt.axis('off')
        if save is not None:
            plt.savefig(save)
        plt.show()