from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _convert_flags(self, fromlist, tolist, flaglist, context, numlines):
    """Makes list of "next" links"""
    toprefix = self._prefix[1]
    next_id = [''] * len(flaglist)
    next_href = [''] * len(flaglist)
    num_chg, in_change = (0, False)
    last = 0
    for i, flag in enumerate(flaglist):
        if flag:
            if not in_change:
                in_change = True
                last = i
                i = max([0, i - numlines])
                next_id[i] = ' id="difflib_chg_%s_%d"' % (toprefix, num_chg)
                num_chg += 1
                next_href[last] = '<a href="#difflib_chg_%s_%d">n</a>' % (toprefix, num_chg)
        else:
            in_change = False
    if not flaglist:
        flaglist = [False]
        next_id = ['']
        next_href = ['']
        last = 0
        if context:
            fromlist = ['<td></td><td>&nbsp;No Differences Found&nbsp;</td>']
            tolist = fromlist
        else:
            fromlist = tolist = ['<td></td><td>&nbsp;Empty File&nbsp;</td>']
    if not flaglist[0]:
        next_href[0] = '<a href="#difflib_chg_%s_0">f</a>' % toprefix
    next_href[last] = '<a href="#difflib_chg_%s_top">t</a>' % toprefix
    return (fromlist, tolist, flaglist, next_href, next_id)