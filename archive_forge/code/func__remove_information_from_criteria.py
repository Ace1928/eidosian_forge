import collections
import itertools
import operator
from .providers import AbstractResolver
from .structs import DirectedGraph, IteratorMapping, build_iter_view
def _remove_information_from_criteria(self, criteria, parents):
    """Remove information from parents of criteria.

        Concretely, removes all values from each criterion's ``information``
        field that have one of ``parents`` as provider of the requirement.

        :param criteria: The criteria to update.
        :param parents: Identifiers for which to remove information from all criteria.
        """
    if not parents:
        return
    for key, criterion in criteria.items():
        criteria[key] = Criterion(criterion.candidates, [information for information in criterion.information if information.parent is None or self._p.identify(information.parent) not in parents], criterion.incompatibilities)