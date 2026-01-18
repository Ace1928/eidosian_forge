import collections
import itertools
import operator
from .providers import AbstractResolver
from .structs import DirectedGraph, IteratorMapping, build_iter_view
def _attempt_to_pin_criterion(self, name):
    criterion = self.state.criteria[name]
    causes = []
    for candidate in criterion.candidates:
        try:
            criteria = self._get_updated_criteria(candidate)
        except RequirementsConflicted as e:
            self._r.rejecting_candidate(e.criterion, candidate)
            causes.append(e.criterion)
            continue
        satisfied = all((self._p.is_satisfied_by(requirement=r, candidate=candidate) for r in criterion.iter_requirement()))
        if not satisfied:
            raise InconsistentCandidate(candidate, criterion)
        self._r.pinning(candidate=candidate)
        self.state.criteria.update(criteria)
        self.state.mapping.pop(name, None)
        self.state.mapping[name] = candidate
        return []
    return causes