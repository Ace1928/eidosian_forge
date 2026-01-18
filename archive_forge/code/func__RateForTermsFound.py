from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.help_search import lookup
def _RateForTermsFound(self):
    """Get a rating based on how many of the searched terms were found."""
    rating = 1.0
    results = self._results.FoundTermsMap()
    for term in self._terms:
        if term not in results:
            rating *= self._NOT_FOUND_MULTIPLIER
    return rating