from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _collect_lines(self, diffs):
    """Collects mdiff output into separate lists

        Before storing the mdiff from/to data into a list, it is converted
        into a single line of text with HTML markup.
        """
    fromlist, tolist, flaglist = ([], [], [])
    for fromdata, todata, flag in diffs:
        try:
            fromlist.append(self._format_line(0, flag, *fromdata))
            tolist.append(self._format_line(1, flag, *todata))
        except TypeError:
            fromlist.append(None)
            tolist.append(None)
        flaglist.append(flag)
    return (fromlist, tolist, flaglist)