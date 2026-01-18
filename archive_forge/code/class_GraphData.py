from . import schema
from .jsonutil import get_column
from .search import Search
class GraphData(object):

    def __init__(self, interface):
        self._intf = interface
        self._struct = interface._struct

    def datatypes(self, pattern='*'):
        graph = nx.DiGraph()
        graph.add_node('datatypes')
        graph.labels = {'datatypes': 'datatypes'}
        graph.weights = {'datatypes': 100.0}
        datatypes = self._intf.inspect.datatypes(pattern)
        namespaces = set([dat.split(':')[0] for dat in datatypes])
        for ns in namespaces:
            graph.add_edge('datatypes', ns)
            graph.weights[ns] = 70.0
            for dat in datatypes:
                if dat.startswith(ns):
                    graph.add_edge(ns, dat)
                    graph.weights[dat] = 40.0
        return graph

    def rest_resource(self, name):
        resource_types = self._intf.inspect._resource_types(name)
        graph = nx.DiGraph()
        graph.add_node(name)
        graph.labels = {name: name}
        graph.weights = {name: 100.0}
        namespaces = set([exp.split(':')[0] for exp in resource_types])
        for ns in namespaces:
            graph.add_edge(name, ns)
            graph.weights[ns] = 70.0
            for exp in resource_types:
                if exp.startswith(ns):
                    graph.add_edge(ns, exp)
                    graph.weights[exp] = 40.0
        return graph

    def field_values(self, field_name):
        search_tbl = Search(field_name.split('/')[0], [field_name], self._intf)
        criteria = [('%s/ID' % field_name.split('/')[0], 'LIKE', '%'), 'AND']
        dist = {}
        for entry in search_tbl.where(criteria):
            for val in entry.values():
                dist.setdefault(val, 1.0)
                dist[val] += 1
        graph = nx.Graph()
        graph.add_node(field_name)
        graph.weights = dist
        graph.weights[field_name] = 100.0
        for val in dist.keys():
            graph.add_edge(field_name, val)
        return graph

    def architecture(self, with_datatypes=True):
        graph = nx.DiGraph()
        graph.add_node('projects')
        graph.labels = {'projects': 'projects'}
        graph.weights = {'projects': 100.0}

        def traverse(lkw, as_lkw):
            for key in schema.resources_tree[lkw]:
                as_key = '%s_%s' % (as_lkw, key)
                weight = (1 - len(as_key * 2) / 100.0) * 100
                graph.add_edge(as_lkw, as_key)
                graph.labels[as_key] = key
                graph.weights[as_key] = weight
                if with_datatypes:
                    for uri in self._struct.keys():
                        if uri.split('/')[-2] == key:
                            datatype = self._struct[uri]
                            graph.add_edge(as_key, datatype)
                            graph.weights[datatype] = 10
                            graph.labels[datatype] = datatype
                traverse(key, as_key)
        traverse('projects', 'projects')
        return graph