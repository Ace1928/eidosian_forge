from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class FeatureIncrementalChart(IncrementalChart, FeatureChart):

    def select(self, end, **restrictions):
        edgelist = self._edgelists[end]
        if restrictions == {}:
            return iter(edgelist)
        restr_keys = sorted(restrictions.keys())
        restr_keys = tuple(restr_keys)
        if restr_keys not in self._indexes:
            self._add_index(restr_keys)
        vals = tuple((self._get_type_if_possible(restrictions[key]) for key in restr_keys))
        return iter(self._indexes[restr_keys][end].get(vals, []))

    def _add_index(self, restr_keys):
        for key in restr_keys:
            if not hasattr(EdgeI, key):
                raise ValueError('Bad restriction: %s' % key)
        index = self._indexes[restr_keys] = tuple(({} for x in self._positions()))
        for end, edgelist in enumerate(self._edgelists):
            this_index = index[end]
            for edge in edgelist:
                vals = tuple((self._get_type_if_possible(getattr(edge, key)()) for key in restr_keys))
                this_index.setdefault(vals, []).append(edge)

    def _register_with_indexes(self, edge):
        end = edge.end()
        for restr_keys, index in self._indexes.items():
            vals = tuple((self._get_type_if_possible(getattr(edge, key)()) for key in restr_keys))
            index[end].setdefault(vals, []).append(edge)