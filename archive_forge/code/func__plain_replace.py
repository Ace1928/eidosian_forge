from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _plain_replace(self, a, alo, ahi, b, blo, bhi):
    assert alo < ahi and blo < bhi
    if bhi - blo < ahi - alo:
        first = self._dump('+', b, blo, bhi)
        second = self._dump('-', a, alo, ahi)
    else:
        first = self._dump('-', a, alo, ahi)
        second = self._dump('+', b, blo, bhi)
    for g in (first, second):
        yield from g