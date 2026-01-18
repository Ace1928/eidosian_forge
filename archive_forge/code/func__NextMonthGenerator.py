from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import calendar
import datetime
from . import groc
def _NextMonthGenerator(self, start, matches):
    """Creates a generator that produces results from the set 'matches'.

    Matches must be >= 'start'. If none match, the wrap counter is incremented,
    and the result set is reset to the full set. Yields a 2-tuple of (match,
    wrapcount).

    Arguments:
      start: first set of matches will be >= this value (an int)
      matches: the set of potential matches (a sequence of ints)

    Yields:
      a two-tuple of (match, wrap counter). match is an int in range (1-12),
      wrapcount is a int indicating how many times we've wrapped around.
    """
    potential = matches = sorted(matches)
    after = start - 1
    wrapcount = 0
    while True:
        potential = [x for x in potential if x > after]
        if not potential:
            wrapcount += 1
            potential = matches
        after = potential[0]
        yield (after, wrapcount)