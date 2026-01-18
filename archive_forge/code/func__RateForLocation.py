from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.help_search import lookup
def _RateForLocation(self):
    """Get a rating based on locations of results."""
    rating = 1.0
    locations = self._results.FoundTermsMap().values()
    for location in locations:
        if location == lookup.NAME:
            rating *= self._COMMAND_NAME_MULTIPLIER
        elif location == lookup.PATH:
            rating *= self._PATH_MULTIPLIER
        elif location.split(lookup.DOT)[0] in [lookup.FLAGS, lookup.POSITIONALS] and location.split(lookup.DOT)[-1] == lookup.NAME:
            rating *= self._ARG_NAME_MULTIPLIER
        else:
            rating *= self._DEFAULT_MULTIPLIER
    return rating