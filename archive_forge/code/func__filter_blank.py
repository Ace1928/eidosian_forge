from __future__ import absolute_import
from functools import reduce
import six
def _filter_blank(i):
    for s in i:
        if s.strip():
            yield s