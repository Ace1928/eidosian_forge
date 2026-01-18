import collections
import itertools
import operator
from .providers import AbstractResolver
from .structs import DirectedGraph, IteratorMapping, build_iter_view
def _patch_criteria():
    for k, incompatibilities in incompatibilities_from_broken:
        if not incompatibilities:
            continue
        try:
            criterion = self.state.criteria[k]
        except KeyError:
            continue
        matches = self._p.find_matches(identifier=k, requirements=IteratorMapping(self.state.criteria, operator.methodcaller('iter_requirement')), incompatibilities=IteratorMapping(self.state.criteria, operator.attrgetter('incompatibilities'), {k: incompatibilities}))
        candidates = build_iter_view(matches)
        if not candidates:
            return False
        incompatibilities.extend(criterion.incompatibilities)
        self.state.criteria[k] = Criterion(candidates=candidates, information=list(criterion.information), incompatibilities=incompatibilities)
    return True