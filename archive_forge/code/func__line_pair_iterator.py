from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _line_pair_iterator():
    """Yields from/to lines of text with a change indication.

        This function is an iterator.  It itself pulls lines from the line
        iterator.  Its difference from that iterator is that this function
        always yields a pair of from/to text lines (with the change
        indication).  If necessary it will collect single from/to lines
        until it has a matching pair from/to pair to yield.

        Note, this function is purposefully not defined at the module scope so
        that data it needs from its parent function (within whose context it
        is defined) does not need to be of module scope.
        """
    line_iterator = _line_iterator()
    fromlines, tolines = ([], [])
    while True:
        while len(fromlines) == 0 or len(tolines) == 0:
            try:
                from_line, to_line, found_diff = next(line_iterator)
            except StopIteration:
                return
            if from_line is not None:
                fromlines.append((from_line, found_diff))
            if to_line is not None:
                tolines.append((to_line, found_diff))
        from_line, fromDiff = fromlines.pop(0)
        to_line, to_diff = tolines.pop(0)
        yield (from_line, to_line, fromDiff or to_diff)