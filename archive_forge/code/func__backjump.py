import collections
import itertools
import operator
from .providers import AbstractResolver
from .structs import DirectedGraph, IteratorMapping, build_iter_view
def _backjump(self, causes):
    """Perform backjumping.

        When we enter here, the stack is like this::

            [ state Z ]
            [ state Y ]
            [ state X ]
            .... earlier states are irrelevant.

        1. No pins worked for Z, so it does not have a pin.
        2. We want to reset state Y to unpinned, and pin another candidate.
        3. State X holds what state Y was before the pin, but does not
           have the incompatibility information gathered in state Y.

        Each iteration of the loop will:

        1.  Identify Z. The incompatibility is not always caused by the latest
            state. For example, given three requirements A, B and C, with
            dependencies A1, B1 and C1, where A1 and B1 are incompatible: the
            last state might be related to C, so we want to discard the
            previous state.
        2.  Discard Z.
        3.  Discard Y but remember its incompatibility information gathered
            previously, and the failure we're dealing with right now.
        4.  Push a new state Y' based on X, and apply the incompatibility
            information from Y to Y'.
        5a. If this causes Y' to conflict, we need to backtrack again. Make Y'
            the new Z and go back to step 2.
        5b. If the incompatibilities apply cleanly, end backtracking.
        """
    incompatible_reqs = itertools.chain((c.parent for c in causes if c.parent is not None), (c.requirement for c in causes))
    incompatible_deps = {self._p.identify(r) for r in incompatible_reqs}
    while len(self._states) >= 3:
        del self._states[-1]
        incompatible_state = False
        while not incompatible_state:
            try:
                broken_state = self._states.pop()
                name, candidate = broken_state.mapping.popitem()
            except (IndexError, KeyError):
                raise ResolutionImpossible(causes)
            current_dependencies = {self._p.identify(d) for d in self._p.get_dependencies(candidate)}
            incompatible_state = not current_dependencies.isdisjoint(incompatible_deps)
        incompatibilities_from_broken = [(k, list(v.incompatibilities)) for k, v in broken_state.criteria.items()]
        incompatibilities_from_broken.append((name, [candidate]))

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
        self._push_new_state()
        success = _patch_criteria()
        if success:
            return True
    return False