from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _make_prefix(self):
    """Create unique anchor prefixes"""
    fromprefix = 'from%d_' % HtmlDiff._default_prefix
    toprefix = 'to%d_' % HtmlDiff._default_prefix
    HtmlDiff._default_prefix += 1
    self._prefix = [fromprefix, toprefix]