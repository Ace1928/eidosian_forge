from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.help_search import lookup
def Rate(self):
    """Produce a simple relevance rating for a set of command search results.

    Returns a float in the range (0, 1]. For each term that's found, the rating
    is multiplied by a number reflecting how "important" its location is, with
    command name being the most and flag or positional names being the second
    most important, as well as by how many of the search terms were found.

    Commands are also penalized if duplicate results in a higher release track
    were found.

    Returns:
      rating: float, the rating of the results.
    """
    rating = 1.0
    rating *= self._RateForLocation()
    rating *= self._RateForTermsFound()
    return rating