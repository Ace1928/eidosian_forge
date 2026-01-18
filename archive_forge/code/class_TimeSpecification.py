from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import calendar
import datetime
from . import groc
class TimeSpecification(object):
    """Base class for time specifications."""

    def GetMatches(self, start, n):
        """Returns the next n times that match the schedule, starting at time start.

    Arguments:
      start: a datetime to start from. Matches will start from after this time.
      n:     the number of matching times to return

    Returns:
      a list of n datetime objects
    """
        out = []
        while len(out) < n:
            start = self.GetMatch(start)
            out.append(start)
        return out

    def GetMatch(self, start):
        """Returns the next match after time start.

    Must be implemented in subclasses.

    Arguments:
      start: a datetime to start from. Matches will start from after this time.
        This may be in any pytz time zone, or it may be timezone-naive
        (interpreted as UTC).

    Returns:
      a datetime object in the timezone of the input 'start'
    """
        raise NotImplementedError