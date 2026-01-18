import collections
import itertools
import operator
from .providers import AbstractResolver
from .structs import DirectedGraph, IteratorMapping, build_iter_view
def _get_updated_criteria(self, candidate):
    criteria = self.state.criteria.copy()
    for requirement in self._p.get_dependencies(candidate=candidate):
        self._add_to_criteria(criteria, requirement, parent=candidate)
    return criteria