from time import perf_counter
from nltk.featstruct import TYPE, FeatStruct, find_variables, unify
from nltk.grammar import (
from nltk.parse.chart import (
from nltk.sem import logic
from nltk.tree import Tree
class FeatureChart(Chart):
    """
    A Chart for feature grammars.
    :see: ``Chart`` for more information.
    """

    def select(self, **restrictions):
        """
        Returns an iterator over the edges in this chart.
        See ``Chart.select`` for more information about the
        ``restrictions`` on the edges.
        """
        if restrictions == {}:
            return iter(self._edges)
        restr_keys = sorted(restrictions.keys())
        restr_keys = tuple(restr_keys)
        if restr_keys not in self._indexes:
            self._add_index(restr_keys)
        vals = tuple((self._get_type_if_possible(restrictions[key]) for key in restr_keys))
        return iter(self._indexes[restr_keys].get(vals, []))

    def _add_index(self, restr_keys):
        """
        A helper function for ``select``, which creates a new index for
        a given set of attributes (aka restriction keys).
        """
        for key in restr_keys:
            if not hasattr(EdgeI, key):
                raise ValueError('Bad restriction: %s' % key)
        index = self._indexes[restr_keys] = {}
        for edge in self._edges:
            vals = tuple((self._get_type_if_possible(getattr(edge, key)()) for key in restr_keys))
            index.setdefault(vals, []).append(edge)

    def _register_with_indexes(self, edge):
        """
        A helper function for ``insert``, which registers the new
        edge with all existing indexes.
        """
        for restr_keys, index in self._indexes.items():
            vals = tuple((self._get_type_if_possible(getattr(edge, key)()) for key in restr_keys))
            index.setdefault(vals, []).append(edge)

    def _get_type_if_possible(self, item):
        """
        Helper function which returns the ``TYPE`` feature of the ``item``,
        if it exists, otherwise it returns the ``item`` itself
        """
        if isinstance(item, dict) and TYPE in item:
            return item[TYPE]
        else:
            return item

    def parses(self, start, tree_class=Tree):
        for edge in self.select(start=0, end=self._num_leaves):
            if isinstance(edge, FeatureTreeEdge) and edge.lhs()[TYPE] == start[TYPE] and unify(edge.lhs(), start, rename_vars=True):
                yield from self.trees(edge, complete=True, tree_class=tree_class)