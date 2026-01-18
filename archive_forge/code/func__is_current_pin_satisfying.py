import collections
import itertools
import operator
from .providers import AbstractResolver
from .structs import DirectedGraph, IteratorMapping, build_iter_view
def _is_current_pin_satisfying(self, name, criterion):
    try:
        current_pin = self.state.mapping[name]
    except KeyError:
        return False
    return all((self._p.is_satisfied_by(requirement=r, candidate=current_pin) for r in criterion.iter_requirement()))